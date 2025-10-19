# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import argparse
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from config import os, name_path, data_file_name

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择爬取数据: 双色球/大乐透/六合彩(lhc)")
try:
    # Avoid breaking imports when other scripts pass unrelated CLI flags
    args, _unknown = parser.parse_known_args()
except SystemExit:
    # In import contexts, ignore parser errors
    class _A:  # minimal fallback
        name = "ssq"
    args = _A()


def get_url(name):
    """
    :param name: 玩法名称
    :return:
    """
    if name in ("ssq", "dlt"):
        url = "https://datachart.500.com/{}/history/".format(name)
        path = "newinc/history.php?start={}&end="
        return url, path
    elif name == "lhc":
        # 新澳门六合彩（2025年）数据页面
        # 站点提供按年份展示完整记录
        return "https://kj.123720c.com/kj/", "?year={}"  # 使用 year 参数
    else:
        raise ValueError("未知玩法: {}".format(name))


def get_current_number(name):
    """ 获取最新一期期号
    :return: str
    """
    if name in ("ssq", "dlt"):
        url, _ = get_url(name)
        r = requests.get("{}{}".format(url, "history.shtml"), verify=False)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "lxml")
        current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
        return current_num
    elif name == "lhc":
        base, path = get_url(name)
        # 默认抓取 2025 年
        year = 2025
        r = requests.get(f"{base}{path.format(year)}", timeout=20)
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "lxml")
        tit = soup.find("div", class_="kj-tit")
        if tit is None:
            raise RuntimeError("未能解析最新期号，目标页面结构可能变化")
        # 例: "六合彩开奖记录 2025年10月18日 第291期"
        m = re.search(r"第\s*(\d+)\s*期", tit.get_text(strip=True))
        if not m:
            raise RuntimeError("未匹配到期号")
        return m.group(1)
    else:
        raise ValueError("未知玩法: {}".format(name))


def spider(name, start, end, mode):
    """ 爬取历史数据
    :param name 玩法
    :param start 开始一期
    :param end 最近一期/结束期
    :param mode 模式，train：训练模式，predict：预测模式（训练模式会保持文件）
    :return: DataFrame
    """
    if name in ("ssq", "dlt"):
        url, path = get_url(name)
        url = "{}{}{}".format(url, path.format(start), end)
        r = requests.get(url=url, verify=False)
        r.encoding = "gb2312"
        soup = BeautifulSoup(r.text, "lxml")
        trs = soup.find("tbody", attrs={"id": "tdata"}).find_all("tr")
        data = []
        for tr in trs:
            item = dict()
            if name == "ssq":
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                for i in range(6):
                    item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
                item[u"蓝球"] = tr.find_all("td")[7].get_text().strip()
                data.append(item)
            elif name == "dlt":
                item[u"期数"] = tr.find_all("td")[0].get_text().strip()
                for i in range(5):
                    item[u"红球_{}".format(i+1)] = tr.find_all("td")[i+1].get_text().strip()
                for j in range(2):
                    item[u"蓝球_{}".format(j+1)] = tr.find_all("td")[6+j].get_text().strip()
                data.append(item)
        df = pd.DataFrame(data)
        if mode == "train":
            df.to_csv("{}{}".format(name_path[name]["path"], data_file_name), encoding="utf-8")
        return df
    elif name == "lhc":
        # 仅支持按年抓取，过滤期号范围
        base, path = get_url(name)
        year = 2025
        r = requests.get(f"{base}{path.format(year)}", timeout=20)
        r.encoding = r.apparent_encoding or "utf-8"
        soup = BeautifulSoup(r.text, "lxml")
        data = []
        # 页面由若干 (kj-tit, kj-box) 组成
        for tit in soup.find_all("div", class_="kj-tit"):
            txt = tit.get_text(strip=True)
            m = re.search(r"第\s*(\d+)\s*期", txt)
            if not m:
                continue
            issue = m.group(1)
            try:
                issue_i = int(issue)
            except Exception:
                continue
            if issue_i < int(start) or issue_i > int(end):
                continue
            box = tit.find_next_sibling("div", class_="kj-box")
            if box is None:
                continue
            dts = [dt.get_text(strip=True) for dt in box.select("dl dt")]
            # 过滤空白和加号占位
            nums = [n for n in dts if n and n.isdigit()]
            if len(nums) < 7:
                # 有时数字被分散在多个 li 中，备选策略：从 class=ball-* 精确提取
                nums = [dt.get_text(strip=True) for dt in box.select("dt[class^=ball-]") if dt.get_text(strip=True).isdigit()]
            if len(nums) >= 7:
                item = {u"期数": f"{issue_i:03d}"}
                for i in range(6):
                    item[u"红球_{}".format(i + 1)] = int(nums[i])
                item[u"蓝球"] = int(nums[6])  # 特码
                data.append(item)
        # 期号升序排序
        data = sorted(data, key=lambda x: int(x["期数"]))
        df = pd.DataFrame(data)
        if mode == "train":
            out_dir = name_path[name]["path"]
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            df.to_csv("{}{}".format(out_dir, data_file_name), encoding="utf-8")
        return df
    else:
        logger.warning("抱歉，没有找到数据源！")
        return pd.DataFrame([])


def run(name):
    """
    :param name: 玩法名称
    :return:
    """
    current_number = get_current_number(name)
    logger.info("【{}】最新一期期号：{}".format(name_path[name]["name"], current_number))
    logger.info("正在获取【{}】数据。。。".format(name_path[name]["name"]))
    if not os.path.exists(name_path[name]["path"]):
        os.makedirs(name_path[name]["path"])
    # 对于六合彩按需求抓取 001-291 期
    if name == "lhc":
        data = spider(name, 1, 291, "train")
        logger.info("已抓取六合彩 2025 年 001-291 期，共 {} 期".format(len(data)))
    else:
        data = spider(name, 1, current_number, "train")
    if "data" in os.listdir(os.getcwd()):
        logger.info("【{}】数据准备就绪，共{}期, 下一步可训练模型...".format(name_path[name]["name"], len(data)))
    else:
        logger.error("数据文件不存在！")


if __name__ == "__main__":
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(name=args.name)
