import requests
import json
import time


def fetch_all_fund_data(fund_code="009608", target_count=1280):
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    all_data = []
    page_size = 20
    page_index = 1

    # 计算最大尝试页数，防止死循环
    max_pages = target_count // page_size + 1

    # --- 关键：请求头必须完整 ---
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://fundf10.eastmoney.com/",
        "Host": "api.fund.eastmoney.com"
    }

    while page_index <= max_pages:
        params = {
            "fundCode": fund_code,
            "pageIndex": page_index,
            "pageSize": page_size,
            # "callback": "jQuery..." 这里坚决不要加
        }

        try:
            print(f"正在请求第 {page_index} 页...")
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response_text = response.text

            # --- 1. 清洗 JSONP 包裹 ---
            # 标准 JSONP 格式: jQueryXXXXXX({...})
            if "(" in response_text and ")" in response_text:
                json_str = response_text[response_text.find("(")+1 : response_text.rfind(")")]
            else:
                # 如果已经是 JSON，直接使用
                json_str = response_text

            # --- 2. 尝试解析 JSON ---
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                print(f"❌ 第 {page_index} 页解析 JSON 失败，内容为：{response_text[:100]}...")  # 打印前100字看是什么
                break

            # --- 3. 提取核心数据列表 ---
            # 根据你之前的截图，结构是 data["Data"]["LSJZList"]
            lsjz_list = data.get("Data", {}).get("LSJZList", [])

            # 如果列表为空，说明数据已经抓完，提前结束
            if not lsjz_list:
                print(f"✅ 第 {page_index} 页无数据，抓取结束。")
                break

            all_data.extend(lsjz_list)
            print(f"  ➡️  已获取 {len(lsjz_list)} 条，累计 {len(all_data)} 条")

            # --- 4. 判断是否达到目标 ---
            if len(all_data) >= target_count:
                print(f"🎉 已达到目标数量 {target_count} 条，停止。")
                break

            # --- 5. 防止被封 IP，加个延迟 ---
            time.sleep(0.5)

        except Exception as e:
            print(f"⚠️  请求第 {page_index} 页时发生网络错误: {e}")
            break

        page_index += 1

    return all_data


# --- 执行程序 ---
if __name__ == "__main__":
    # 你想查询的基金代码
    fund_code = "009608"

    # 调用函数，目标获取 1280 条
    results = fetch_all_fund_data(fund_code, target_count=1280)

    print(f"\n最终结果：共获取到 {len(results)} 条数据")

    # 保存到文件
    with open("fund_data.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
