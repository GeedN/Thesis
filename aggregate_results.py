from pathlib import Path
import pandas as pd # type: ignore
    
def process_df(df):
    df = df.copy()
    df["target"] = df["target"].astype(str).str.lower().str.strip()
    df["result"] = df["result"].astype(str).str.lower().str.strip()
    return df

def resolve_uncertain(row, tiebreaker): #turns uncertains to corrects or incorrects based on optimism or pessimism
    if row["result"] != "uncertain":
        return row["result"]
    if tiebreaker == "uncertain":
        return "uncertain"
    if tiebreaker == row["target"]:
        return "correct"
    else:
        return "incorrect"

def construct_preds(df):
    preds = []
    for target, result in zip(df["target"],df["result"]):
        if result == "uncertain" or result == "na":
            preds.append(None)
        elif target == "true" and result == "correct":
            preds.append(1)
        elif target == "true" and result == "incorrect":
            preds.append(0)
        elif target == "false" and result == "correct":
            preds.append(0)
        elif target == "false" and result == "incorrect":
            preds.append(1)

        else:
            raise ValueError(f"Impossible state reached: Target {target} ; Result {result}")
    return preds

if __name__ == "__main__":
    raw_results = Path("./results/raw/")
    agg_results = []

    for file in raw_results.glob("*.csv"):
        result_dict = {}
        # files ar e named: model_params_dataset_framing
        model, params, dataset, framing = file.stem.lower().split("_")

        size = "large" if float(params[:-1])>5 else "small" #for replication: size assumes last letter is the only text in the parameter info
        instruct = 0

        model_id = f"{model}_{params}"
        if "qwen2.5" in model: # as only and all Qwen 2.5 (7b and 1.5b parameters) are instruct-tuned models in this
            model_id = model_id + "_instruct"
            instruct = 1
        df = pd.read_csv(file)
        df = process_df(df)
        df["pred"] = construct_preds(df)

        #for replication: following is optional additional info (pessimism and optimism defaults)
        # df_opt = df.copy()
        # df_opt("result") = df_opt.apply(resolve_uncertain, axis=1, tiebreaker = "true")
        # df["pred_opt"] = construct_preds(df_opt)
        # df_pes = df.copy()
        # df_pes("result") = df_pes.apply(resolve_uncertain, axis=1, tiebreaker = "false")
        # df["pred_pes"] = construct_preds(df_pes)

        for idx, row in df.iterrows():
            target = 1 if row["target"] == "true" else 0
            valid = row["result"] not in ["na", "uncertain"] #to calculate coverage
            abstain = row["result"] == "uncertain" 
            bin_result = int(row["result"] == "correct") if valid else None #binary result for evaluation

            agg_results.append({
                "model_id": model_id,
                "size": size,
                "instruction": instruct,
                "dataset": dataset,
                "framing": framing,
                "item_id": f"{dataset}_{idx}",
                "target": target,
                "prediction": row["pred"],
                "raw_result": row["result"],
                "bin_result": bin_result,
                #uncomment for pos/neg
                # "pred_opt": row["pred_opt"],
                # "opt_result": int(row["pred_opt"] == row["target"]) if row["pred_opt"],
                # "pred_pes": row["pred_pes"],
                # "pes_result": int(row["pred_pes"] == row["target"]),
                "valid": int(valid),
                "abstain": int(abstain)
            })


    agg_df = pd.DataFrame(agg_results)
    agg_df["effective_accuracy"] = agg_df["bin_result"].fillna(0) #useful for later calculations

    print(agg_df.head())

    agg_df.to_csv("results/results.csv", index=False)