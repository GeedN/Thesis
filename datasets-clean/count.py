import pandas as pd

paths = ["./clean_Liar_test.csv", "clean_Truthseeker.csv"]
false_count = 0
true_count = 0
for file in paths:
    df = pd.read_csv(file)
    for t in df["target"]:
        if t == True:
            true_count += 1
        elif t == False:
            false_count += 1
        else:
            print("invalid:", t)

print(f"true: {true_count}\nfalse: {false_count}")
