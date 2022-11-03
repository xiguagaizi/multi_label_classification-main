
import pandas as pd


def xlsx_to_csv_pd():
    data_xls = pd.read_excel('./model/out.xlsx', index_col=0)  # 输入xlsx文件名
    data_xls.to_csv('./model/out.csv', encoding='utf-8')  # 输出csv文件名


if __name__ == '__main__':
    xlsx_to_csv_pd()
