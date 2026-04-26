import json
import csv
import os


def convert_json_to_csv():
    """
    读取指定路径的 JSON 文件，提取 DWJZ、FSRQ 和 JZZZL 字段，并保存为 CSV。
    """
    # 固定 JSON 文件路径
    json_file_path = r"/other_tests/009608_fund_data.json"

    # 设定输出 CSV 文件路径（与 JSON 文件同目录）
    csv_file_path = os.path.join(os.path.dirname(json_file_path), "009608_fund_data.csv")

    try:
        # 1. 读取 JSON 文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # 如果数据是单个对象，转换为列表
            if isinstance(data, dict):
                data = [data]

        # 2. 提取指定字段
        extracted_data = []
        for item in data:
            dwjz = item.get('DWJZ', '')
            fsrq = item.get('FSRQ', '')
            jzzzl = item.get('JZZZL', '')

            extracted_data.append({
                '单位净值': dwjz,
                '净值日期': fsrq,
                '净值日增长率': jzzzl
            })

        # 3. 写入 CSV 文件
        if extracted_data:
            fieldnames = ['单位净值', '净值日期', '净值日增长率']

            with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(extracted_data)

            print(f"成功：已提取 {len(extracted_data)} 条记录并保存到 {csv_file_path}")
        else:
            print("警告：未在 JSON 文件中找到数据。")

    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
    except json.JSONDecodeError as e:
        print(f"错误：JSON 解析失败 - {e}")
    except Exception as e:
        print(f"发生未知错误：{e}")


# --- 主程序入口 ---
if __name__ == "__main__":
    convert_json_to_csv()