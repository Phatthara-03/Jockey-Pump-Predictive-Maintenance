import pandas as pd
import re

input_path = r"A_Jockey pump_M1A_2925__Jun24.txt"
output_path = r"Jockey_cleaned.csv"

data = []

with open(input_path, "r", encoding="utf-8", errors="ignore") as file:
    lines = file.readlines()

for line in lines:
    line = line.strip()

    if "Time" in line or "---" in line or line == "":
        continue

    numbers = re.findall(r"-?\d+\.\d+|-?\.\d+", line)
    numbers = [float(x) for x in numbers]

    # ถ้าไม่ครบคู่ ข้าม
    if len(numbers) % 2 != 0:
        continue

    for i in range(0, len(numbers) - 1, 2):
        data.append([numbers[i], numbers[i + 1]])

df = pd.DataFrame(data, columns=["Time_ms", "Amplitude"])
df = df.sort_values("Time_ms").reset_index(drop=True)

df.to_csv(output_path, index=False)

print("✅ จัดรูปแบบเสร็จแล้ว")
print("จำนวนข้อมูลทั้งหมด:", len(df))