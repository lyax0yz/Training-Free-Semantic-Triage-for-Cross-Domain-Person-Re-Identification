import argparse
import csv
import json

import numpy as np
from torchreid.semantic import (
   AttributeWeight,
   RankedCandidate,
   SemanticArbitrationConfig,
   SymbolRegistry,
   arbitrate_relative_ambiguity,
)


ATTRS = ("upper_colour", "lower_colour")


def norm(x):
   x = np.asarray(x, dtype=np.float32)
   return x / np.linalg.norm(x, axis=1, keepdims=True)


def attrs(item, reg):
   source = (item or {}).get("attributes", {})
   output = {}
   for name in ATTRS:
      record = source.get(name, {})
      value = record.get("value") if isinstance(record, dict) else record
      confidence = (
         float(record.get("confidence", 0)) if isinstance(record, dict) else 1.0
      )
      output[name] = (
         reg.normalize(name, value) if confidence >= 0.25 else "unknown"
      )
   return output


def dist(values):
   if not values:
      return {"count": 0}

   array = np.asarray(values, float)
   return {
      "count": len(values),
      "mean": float(array.mean()),
      "median": float(np.median(array)),
      "q25": float(np.quantile(array, 0.25)),
      "q75": float(np.quantile(array, 0.75)),
   }

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument("--artifact", required=True)
   parser.add_argument("--attribute-cache", required=True)
   parser.add_argument("--output-json", required=True)
   parser.add_argument("--output-csv", required=True)
   args = parser.parse_args()

   with np.load(args.artifact, allow_pickle=False) as data:
      query_features = norm(data["query_features"])
      gallery_features = norm(data["gallery_features"])
      query_pids, gallery_pids = data["query_pids"], data["gallery_pids"]
      query_camids, gallery_camids = data["query_camids"], data["gallery_camids"]
      query_paths = data["query_impaths"].astype(str)
      gallery_paths = data["gallery_impaths"].astype(str)

   with open(args.attribute_cache, encoding="utf8") as file:
      cache = json.load(file)["images"]

   registry = SymbolRegistry()
   config = SemanticArbitrationConfig(
      lambda_sem=0.05,
      normalize_adjustment=True,
      weights={name: AttributeWeight(1, 1) for name in ATTRS},
   )
   rows = []
   agreement = {
      key: {"true": [0, 0], "false": [0, 0]} for key in ("D", "E", "F")
   }
   gaps = {key: [] for key in ("D", "E", "F")}
   adjustments = {
      f"{key}_{label}": []
      for key in ("D", "E", "F")
      for label in ("true", "false")
   }

   for query_index in range(len(query_features)):
      scores = np.clip(
         query_features[query_index].dot(gallery_features.T), -1, 1
      )
      order = np.argsort(-scores, kind="mergesort")
      valid = order[
         ~(
            (gallery_pids[order] == query_pids[query_index])
            & (gallery_camids[order] == query_camids[query_index])
         )
      ]

      if gallery_pids[valid[0]] == query_pids[query_index]:
         continue

      top = valid[:20]
      true_top = np.flatnonzero(gallery_pids[top] == query_pids[query_index])
      prefix = top[scores[top] >= scores[top[0]] - 0.02]
      true_ambiguity = np.flatnonzero(
         gallery_pids[prefix] == query_pids[query_index]
      )
      stage = (
         "B"
         if not len(true_top)
         else "C"
         if not len(true_ambiguity)
         else "D"
      )
      row = {
         "query_index": query_index,
         "query_path": query_paths[query_index],
         "query_pid": int(query_pids[query_index]),
         "baseline_false_path": gallery_paths[valid[0]],
         "baseline_false_pid": int(gallery_pids[valid[0]]),
         "baseline_false_visual_similarity": float(scores[valid[0]]),
         "funnel_stage": stage,
         "true_in_top20": bool(len(true_top)),
         "true_in_ambiguity": bool(len(true_ambiguity)),
      }

      if stage == "D":
         best_index = prefix[true_ambiguity[0]]
         query_attributes = attrs(cache.get(query_paths[query_index]), registry)
         ranking = [
            RankedCandidate(gallery_paths[index], float(scores[index]))
            for index in prefix
         ]
         results, _audit = arbitrate_relative_ambiguity(
            query_attributes,
            ranking,
            lambda path: attrs(cache.get(path), registry),
            config,
            delta=0.02,
            top_k=20,
         )
         reordered = np.asarray(
            [
               next(index for index in prefix if gallery_paths[index] == item["key"])
               for item in results
            ]
         )
         semantic_order = np.concatenate((reordered, valid[len(prefix) :]))
         rank_after = int(np.flatnonzero(semantic_order == best_index)[0]) + 1
         rank_before = int(np.flatnonzero(valid == best_index)[0]) + 1
         corrected = gallery_pids[semantic_order[0]] == query_pids[query_index]
         stage = "E" if corrected else "F"
         result_by_key = {item["key"]: item for item in results}
         true_result = result_by_key[gallery_paths[best_index]]
         false_result = result_by_key[gallery_paths[valid[0]]]

         row.update(
            {
               "funnel_stage": stage,
               "best_true_path": gallery_paths[best_index],
               "best_true_visual_similarity": float(scores[best_index]),
               "best_true_rank_before": rank_before,
               "best_true_rank_after": rank_after,
               "semantic_corrected_rank1": bool(corrected),
               "true_minus_false_visual_gap": float(
                  scores[best_index] - scores[valid[0]]
               ),
               "true_semantic_adjustment": float(
                  true_result["semantic_adjustment"]
               ),
               "false_semantic_adjustment": float(
                  false_result["semantic_adjustment"]
               ),
            }
         )

         true_attributes = attrs(cache.get(gallery_paths[best_index]), registry)
         false_attributes = attrs(cache.get(gallery_paths[valid[0]]), registry)
         for name in ATTRS:
            for label, candidate_attributes in (
               ("true", true_attributes),
               ("false", false_attributes),
            ):
               if (
                  query_attributes[name] != "unknown"
                  and candidate_attributes[name] != "unknown"
               ):
                  agreement["D"][label][1] += 1
                  agreement["D"][label][0] += int(
                     query_attributes[name] == candidate_attributes[name]
                  )
                  agreement[stage][label][1] += 1
                  agreement[stage][label][0] += int(
                     query_attributes[name] == candidate_attributes[name]
                  )

         for key in ("D", stage):
            gaps[key].append(row["true_minus_false_visual_gap"])
            adjustments[f"{key}_true"].append(row["true_semantic_adjustment"])
            adjustments[f"{key}_false"].append(row["false_semantic_adjustment"])

      rows.append(row)

   counts = {
      key: sum(row["funnel_stage"] == key for row in rows)
      for key in ("B", "C", "E", "F")
   }
   total = len(rows)
   d_count = counts["E"] + counts["F"]
   funnel = {
      "A_total_baseline_rank1_failures": total,
      "B_true_absent_top20": counts["B"],
      "C_true_top20_outside_ambiguity": counts["C"],
      "D_true_inside_ambiguity": d_count,
      "E_corrected": counts["E"],
      "F_not_corrected": counts["F"],
   }
   summary = {
      "configuration": {
         "topk": 20,
         "delta": 0.02,
         "attributes": list(ATTRS),
         "confidence_threshold": 0.25,
         "lambda_sem": 0.05,
      },
      "funnel": {
         key: {"count": value, "percentage_of_A": value / total}
         for key, value in funnel.items()
      },
      "D_E_F_statistics": {
         key: {
            "visual_gap": dist(gaps[key]),
            "colour_agreement_rate": {
               label: (
                  agreement[key][label][0] / agreement[key][label][1]
                  if agreement[key][label][1]
                  else None
               )
               for label in ("true", "false")
            },
            "true_semantic_adjustment": dist(adjustments[f"{key}_true"]),
            "false_semantic_adjustment": dist(adjustments[f"{key}_false"]),
         }
         for key in ("D", "E", "F")
      },
   }
   with open(args.output_json, "w", encoding="utf8") as file:
      json.dump(summary, file, indent=2)

   fields = sorted({key for row in rows for key in row})
   with open(args.output_csv, "w", newline="", encoding="utf8") as file:
      writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
      writer.writeheader()
      writer.writerows(rows)


if __name__ == "__main__":
   main()
