# encoding: utf-8

# 人为输入多个数据集字典的处理方法


class automation_eda:
    def __init__(
        self,
        test_size=0.3,
        iv_threshold=0.01,
        iv_upper=None,
        corr_threshold=0.8,
        empty_threshold=0.8,
        target_col="target",
        id_col=None,
        data_path="test_data.csv",
        data_dict=None,
        fillna_value=-9999,
        train_test_split=False,
        select_method="toad",
        xgb_params={
            "max_depth": 4,
            "n_estimators": 30,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.5,
            "reg_alpha": 0.5,
            "gamma": 0.5,
            "min_child_weight": 5,
            "eval_metric": "auc",
            "early_stopping_rounds": 5,
            "random_state": 42,
        },
        show_params={
            "base": True,
            "tag_effect": True,
            "fea_dis": [
                True,
                {"kde": True, "bivar": True, "method": "freq", "nbins": 10},
            ],
            "fea_imp": [True, {"iv": True, "corr": True}],
            "model_fea": [True, {"iv": True, "corr": True, "del": True}],
            "model_res": [True, {"optuna": False, "split_num": 20}],
        },
        report_file_name="自动化探查报告",
        report_title="自动化探查报告",
        save_csv=None,
    ):
        """
        :param test_size: 数据训练集和测试集划分比列, 默认 0.3
        :param iv_threshold: 特征筛选的iv阈值下限, 默认 0.01
        :param iv_upper: 特征筛选的iv阈值上限, 默认 None
        :param corr_threshold: 特征筛选的相关性阈值, 默认 0.8
        :param empty_threshold: 特征筛选的缺失率阈值, 默认 0.8
        :param target_col: 数据集中标签名称（需要根据实际的数据集进行修改）, 默认 'target'
        :param id_col: 数据集id列名称（需要根据实际的数据集进行修改，当不存在id列时为默认''）, 默认 None
        :param data_path: 数据文件路径（需要根据实际情况修改）或者df变量, 默认 'test_data.csv'
        :param data_dict：人为处理后的数据集分类，默认为None，不为None时字典必须包含训练集，例如{"train":train_data}
        :param fillna_value: 缺失值填充值, 默认 -9999
        :param train_test_split: 是否进行训练集和测试集划分, 默认 False
        :param select_method: 特征筛选方法, 默认 'toad'，可选'custom'（iv值和相关性筛选）
        :param xgb_params：XGBoost模型参数，默认xgb_params = {
                                                    "max_depth": 4,
                                                    "n_estimators": 30,
                                                    "learning_rate": 0.05,
                                                    "subsample": 0.8,
                                                    "colsample_bytree": 0.8,
                                                    "lambda": 0.5,
                                                    "alpha": 0.5,
                                                    "gamma": 0.5,
                                                    "min_child_weight": 5,
                                                    "eval_metric": "auc",
                                                    "early_stopping_rounds": 5,
                                                    "random_state": 42,
                                                }
        :param show_params: 报告功能模块展示的自定义参数, 默认 show_params={
            "base": True,
            "tag_effect": True,
            "fea_dis": [True, {"box": True, "bin": True, "method": "freq"}],
            "fea_imp": [True, {"iv": True, "corr": True}],
            "model_fea": [True, {"iv": True, "corr": True, "del": True}],
            "model_res": [True, {"optuna": False, "split_num": 20}],
            }
        :param report_file_name: 最终生成报告文件名称, 默认 '自动化探查报告'
        :param report_title: 报告内部标题名称, 默认 '自动化探查报告'
        :param save_csv：要保存EDA对应的id+特征+target数据时指定保存名称，例如“eda_data.csv”，默认为None
        """
        import pandas as pd
        import time
        import os

        self.test_size = test_size
        self.iv_threshold = iv_threshold
        self.corr_threshold = corr_threshold
        self.empty_threshold = empty_threshold
        self.target_col = target_col
        self.id_col = id_col
        self.data_path = data_path
        self.data_dict = data_dict
        self.fillna_value = fillna_value
        self.train_test_split = train_test_split
        self.select_method = select_method
        self.report_file_name = report_file_name
        self.report_title = report_title
        self.xgb_params = xgb_params
        self.show_params = show_params
        self.iv_upper = iv_upper

        # data_dict不为None时优先data_dict否则以data_path
        if self.data_dict is not None:  # 判断data_dict是否为空
            if "train" not in self.data_dict.keys():
                raise KeyError("data_dict字典中必须指定train数据集！")
            elif len(self.data_dict.keys()) > 3:
                raise ValueError("data_dict字典中不能超过3个数据集！")
            else:
                data_mode = 1
        else:
            data_mode = 0
            if isinstance(self.data_path, pd.DataFrame):
                self.data = self.data_path
            elif os.path.exists(self.data_path) and os.path.isfile(self.data_path):
                self.data = pd.read_csv(self.data_path)  # data_path为正确的文件路径时读取数据
            else:
                raise ValueError("data_path不是正确的文件路径或者pd.DataFrame!")
        self.save_name = save_csv
        self.data_mode = data_mode
        self.start_time = time.time()

    def fit(self):
        import pandas as pd
        import numpy as np
        import sys
        import copy
        import os
        import time
        import datetime
        import toad
        import scorecardpy as sc
        import seaborn as sns
        import sweetviz as sv

        # import pandas_profiling as pp
        import re
        from IPython.display import HTML

        # import optuna

        from sklearn.model_selection import train_test_split
        from sklearn.model_selection import StratifiedKFold
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        from xgboost import XGBClassifier
        from xgboost import plot_importance

        import matplotlib.pyplot as plt
        from yellowbrick.classifier import ROCAUC
        from yellowbrick.model_selection import FeatureImportances
        from yellowbrick.model_selection import validation_curve
        from yellowbrick.model_selection import RFECV

        from io import BytesIO
        from lxml import etree
        import base64

        from all_utils import Base_Func
        from all_utils import Bins_Cut
        from all_utils import Eval_Module
        from all_utils import TT_Compare_Select
        from all_utils import Eda_Func
        from all_utils import Binning
        from all_utils import Cal_Feas_Perf

        plt.rc("font", family="SimHei", size=12)
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
        plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号

        title_num_map = {
            1: "一",
            2: "二",
            3: "三",
            4: "四",
            5: "五",
            6: "六",
            7: "七",
            8: "八",
            9: "九",
        }

        def fig_to_html(buff, fig_title):
            import base64

            plot_data = buff.getvalue()
            imb = base64.b64encode(plot_data)
            ims = imb.decode()
            imd = "data:image/png;base64," + ims
            html_img = "<p>" + str(fig_title) + "</p>"
            html_img = html_img + """<img src="%s">""" % imd + "<br>"
            return html_img

        body_list = []
        body_h2_list = []

        # 检查id列
        if self.data_mode == 1:
            data = self.data_dict["train"]
        else:
            data = self.data
        if self.id_col is not None and len(self.id_col) >= 1:  # 有id列时
            id_series = data[self.id_col]
            data.drop(columns=self.id_col, inplace=True, errors="ignore")
            print(f"删除ID列如下：{self.id_col}")

        # data = data.loc[data[self.target_col].isin([0, 1])]  # 只选取标签为0,1的样本
        data_info, data_info_des, cols_tuple = Base_Func.data_base_info(
            data, miss_thre=self.empty_threshold
        )
        (
            num_cols,
            char_cols,
            num_conti_cols,
            num_cate_cols,
            one_value_cols,
            more_null_cols,
        ) = cols_tuple
        # 标签分布
        target_dis = pd.DataFrame(data[self.target_col].value_counts(dropna=False))
        target_dis[f"{self.target_col}_prop"] = (
            target_dis[self.target_col] / target_dis[self.target_col].sum()
        )
        target_dis = target_dis.round({f"{self.target_col}_prop": 3})

        target_info = ""
        for index, rows in target_dis.iterrows():
            if index == 1:
                target_info = (
                    target_info
                    + f"黑样本<b>{int(rows.loc[self.target_col])}</b>个，占比<b>{rows.loc[f'{self.target_col}_prop']}</b>；"
                )
            else:
                target_info = (
                    target_info
                    + f"白样本<b>{int(rows.loc[self.target_col])}</b>个，占比<b>{rows.loc[f'{self.target_col}_prop']}</b>；"
                )
        target_info = "<p>" + target_info[:-1] + "。</p>"

        target_html = "<br>1.2 数据集标签分布" + target_dis.to_html() + target_info
        data_info_des = (
            "<i>注：本报告关于连续型和类别型特征的分类（kind）定义为，特征取值数量（unique）大于10为连续型（Continuous）特征，\
                 相反为类别型（Categorical）特征。</i><br><br>"
            + data_info_des
            + target_html
        )
        h2_title_num = 1
        if self.show_params["base"]:
            body_list.append(
                "<br>" * 2
                + re.sub(r"<table .*?>", r"<table border=1>", (data_info.to_html()))
                + data_info_des
            )
            body_h2_list.append("一、基础概况分析")
            h2_title_num += 1

        data = data[num_cols]
        # data_with_na = data.copy(deep=True)
        # data.fillna(self.fillna_value, inplace=True)
        target_rate = data[self.target_col].value_counts()[1] / len(data)

        if self.data_mode == 0 and self.train_test_split:
            # 训练集测试集划分
            train, test = train_test_split(
                data,
                stratify=data[self.target_col],
                test_size=self.test_size,
                random_state=42,
                shuffle=True,
            )
            data_dict = {"train": train, "test": test}
            # train_na, test_na = train_test_split(
            #     data_with_na,
            #     test_size=self.test_size,
            #     stratify=data_with_na[self.target_col],
            #     random_state=42,
            # )
            print(f"训练集维度为：{train.shape}， 测试集维度为：{test.shape}。")
        elif self.data_mode == 0 and not self.train_test_split:
            data_dict = {"train": data, "test": data}
        else:
            data_dict = {
                key: self.data_dict[key][num_cols] for key in self.data_dict.keys()
            }
        ################################ 二、标签效果#####################################
        if self.show_params["tag_effect"]:
            eff_html = "<br>对数据集中所有取值数目小于等5个的特征统计命中率、提升度等效果。<br>"
            styled_df = Cal_Feas_Perf.mul_value_perf(
                data.fillna(self.fillna_value),
                target=self.target_col,
            )
            if isinstance(styled_df, pd.io.formats.style.Styler):
                styled_html = styled_df.to_html()
                styled_data = styled_df.data
                feas_num = styled_data["变量名"].nunique()
                more_lift_num = len(
                    pd.DataFrame(styled_data.groupby("变量名")["提升度"].max()).query("提升度>1")
                )
                eff_html += f"数据集自然转化率为{styled_data['自然转化率'][0]}，统计到的标签特征（特征取值小等于5）共{feas_num}个。"
                eff_html += f"<br><i>注：变量取值{self.fillna_value}表示变量在缺失情况下的命中效果。</i>"
                eff_html = eff_html + re.sub(
                    r"<table .*?>", r"<table border=1>", styled_html
                )
            else:
                eff_html += "<i>没有统计到相关特征。</i>"
            body_list.append(eff_html)
            body_h2_list.append(f"{title_num_map[h2_title_num]}、特征标签效果")
            h2_title_num += 1

        ################################ 二、特征分布#####################################
        # 2.1 连续特征在不同标签下的分布
        # 数值连续性特征在不同标签下的分布
        if self.show_params["fea_dis"][0]:
            sub_params = self.show_params["fea_dis"][1]
            sub_title_num = 1
            dis_html = ""
            if sub_params["kde"]:  # 选择显示kde
                dis_html = f"<p>{h2_title_num}.1 连续特征在不同标签下的分布</p>"
                for col in num_conti_cols:
                    buff = Eda_Func.distributions_with_target(
                        data,
                        col,
                        target=self.target_col,
                        is_pdf=True,
                    )
                    plot_data = buff.getvalue()
                    imb = base64.b64encode(plot_data)
                    ims = imb.decode()
                    imd = "data:image/png;base64," + ims
                    dis_html = (
                        dis_html
                        + """<img src="%s" width="1200", height="400">""" % imd
                        + "<br>" * 2
                    )
                sub_title_num += 1

            # 2.2 特征bivar图
            bi_html = ""
            if sub_params["bivar"]:  # 选择显示bivar图
                data_bins = {}
                if sub_params["method"] == "freq":
                    data_bins["train"] = Bins_Cut.cut_bins_fun(
                        data_dict["train"].fillna(self.fillna_value),
                        self.target_col,
                        num_miss_list=self.fillna_value,
                        cha_miss_list="miss",
                        method="freq",
                        nbins=sub_params["nbins"],
                    )
                    for idx_name, idx_data in data_dict.items():
                        if idx_name != "train":
                            data_bins[idx_name] = Bins_Cut.test_bins_fun(
                                idx_data.fillna(self.fillna_value),
                                y_flag=self.target_col,
                                bins_adj=data_bins["train"],
                                is_special=True,
                            )
                else:
                    data_bins["train"] = Binning.woebin(
                        data_dict["train"],
                        y=self.target_col,
                        stop_limit=0.04,
                        count_distr_limit=0.01,
                        max_num_bin=10,
                        method="chimerge",
                    )
                    for idx_name, idx_data in data_dict.items():
                        if idx_name != "train":
                            data_bins[idx_name] = Bins_Cut.test_bins_fun(
                                idx_data,
                                y_flag=self.target_col,
                                bins_adj=data_bins["train"],
                                is_special=False,
                            )

                # 计算data_bins内个字典key的交集
                bivar_list = list(
                    set.intersection(
                        *[set(idx_dict.keys()) for idx_dict in data_bins.values()]
                    )
                )
                bivar_list = list(
                    dict(
                        sorted(
                            {
                                col: data_bins["train"][col]["total_iv"][0]
                                for col in bivar_list
                            }.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )
                    ).keys()
                )
                if len(bivar_list) >= 1:
                    bi_html = TT_Compare_Select.bivar_html(
                        "连续型特征等频分箱效果分析",
                        {"train": data_bins["train"]}
                        if (self.data_mode == 0 and not self.train_test_split)
                        else data_bins,
                        target_rate,
                        bivar_list,
                        is_root=True,
                    )
                    bi_html = (
                        f"<br><br>{h2_title_num}.{sub_title_num} 特征bivar图"
                        + "<br>" * 2
                        + bi_html
                    )
            dis_html = "<br>" + dis_html + bi_html

            body_list.append(dis_html)
            body_h2_list.append(f"{title_num_map[h2_title_num]}、特征分布")
            h2_title_num += 1

        ################################ 三、特征重要性#####################################
        # 判断不进行数据集划分的情况
        if self.show_params["fea_imp"][0]:  # 显示特征重要性模块
            sub_params = self.show_params["fea_imp"][1]
            if self.data_mode == 0 and self.train_test_split:
                train_test_info = f"<br><p>在去除非数值型特征基础上，对数据集按测试集占<b>{self.test_size}</b>的比例进行划分，\
                            训练集维度{data_dict['train'].shape}，测试集维度{data_dict['test'].shape}。<p><br>"
            else:
                train_test_info = (
                    f"<br><p>在去除非数值型特征基础上，数据集维度{data_dict['train'].shape}。<p><br>"
                )

            # 3.1 训练集特征iv值
            # 全量特征iv值
            sub_title_num = 1
            if sub_params["iv"]:  # 显示iv值
                train_iv = pd.DataFrame(
                    toad.quality(
                        data_dict["train"], target=self.target_col, iv_only=True
                    )["iv"]
                )
                train_iv["iv"] = train_iv["iv"].map(lambda x: "%.3f" % x)
                if self.train_test_split:
                    iv_title = "训练集特征IV值"
                else:
                    iv_title = "数据集特征IV值"
                train_test_info = (
                    train_test_info
                    + f"<p>{h2_title_num}.1 {iv_title}</p>"
                    + re.sub(
                        r"<table .*?>",
                        "<table border=1>",
                        train_iv.style.background_gradient(
                            subset=["iv"], cmap="RdYlGn"
                        ).to_html(),
                    )
                    + "<br>"
                )
                sub_title_num += 1
            # 3.2 训练集特征相关性热力图
            train_one_value = (
                data_dict["train"]
                .nunique()[data_dict["train"].nunique().values == 1]
                .index.tolist()
            )
            mul_value_cols = np.setdiff1d(
                data_dict["train"].drop(columns=[self.target_col]).columns,
                train_one_value,
            ).tolist()
            if self.train_test_split:
                corr_title = "训练集特征相关性热力图"
            else:
                corr_title = "数据集特征相关性热力图"
            html_img1 = f"{h2_title_num}.{sub_title_num} {corr_title}"
            if sub_params["corr"]:
                if len(mul_value_cols) >= 2:
                    buff = Eda_Func.sns_corr(
                        data_dict["train"][mul_value_cols],
                        sns_vars={"annot": True, "fmt": ".2f"},
                        is_pdf=True,
                    )
                    html_img1 = fig_to_html(
                        buff, f"{h2_title_num}.{sub_title_num} {corr_title}"
                    )
            corr = data_dict["train"][mul_value_cols].corr()
            for i in range(corr.shape[0]):
                for j in range(i + 1):
                    corr.iloc[i, j] = 0
            corr = corr[abs(corr) > self.corr_threshold]
            corr = corr.unstack().sort_values(ascending=False)
            corr = corr[corr > self.corr_threshold]
            corr = pd.DataFrame(corr).reset_index()
            corr.columns = ["特征1", "特征2", "相关系数"]
            train_test_info = train_test_info + html_img1 + corr.to_html()

            body_list.append(train_test_info)
            body_h2_list.append(f"{title_num_map[h2_title_num]}、特征重要性")
            h2_title_num += 1

        ################################ 四、入模特征筛选#####################################

        del_cols = list(set(char_cols + one_value_cols + more_null_cols))
        # print(len(del_cols))
        select_cols = np.setdiff1d(data.columns, del_cols).tolist()
        if self.select_method == "toad":
            select, drop_dict = toad.select(
                data_dict["train"][select_cols],
                target=self.target_col,
                iv=self.iv_threshold,
                corr=self.corr_threshold,
                return_drop=True,
            )
        else:
            select, drop_dict = Eda_Func.filter_features(
                data_dict["train"][select_cols],
                target=self.target_col,
                iv_threshold=self.iv_threshold,
                corr_threshold=self.corr_threshold,
                empty_threshold=self.empty_threshold,
                return_drop=True,
            )

        # 整理toad删除的特征
        del_from_toad = pd.DataFrame()
        for col, reason in zip(
            ["iv", "corr", "empty"],
            [
                f"iv值小于{self.iv_threshold}",
                f"相关性大于{self.corr_threshold}",
                f"缺失率大于等于{self.empty_threshold}",
            ],
        ):
            idx_df = pd.DataFrame()
            idx_df["特征"] = drop_dict.get(col, np.array([]))
            idx_df["删除原因"] = reason
            del_from_toad = pd.concat(
                [del_from_toad, idx_df], axis=0, ignore_index=True
            )

        # 整理被删除的特征（所有非入模特征）
        # 1. 非数值型
        char_cols_df = pd.DataFrame()
        char_cols_df["特征"] = char_cols
        char_cols_df["删除原因"] = "非数值型"

        # 2. 只有唯一取值
        one_value_df = pd.DataFrame()
        one_value_df["特征"] = one_value_cols
        one_value_df["删除原因"] = "只有唯一取值"

        # 3. 缺失率大于等于80%
        more_null_df = pd.DataFrame()
        more_null_df["特征"] = more_null_cols
        more_null_df["删除原因"] = f"缺失率大于等于{str(int(self.empty_threshold*100))+'%'}"

        # 6. iv值大于等于阈值上限
        iv_than_cols = []
        try:
            train_iv = toad.quality(select, target=self.target_col, iv_only=True)
            if str(self.iv_upper).isdigit():
                iv_than_cols = train_iv[train_iv["iv"] >= self.iv_upper].index.tolist()
        except KeyError:
            pass
        iv_than_df = pd.DataFrame()
        iv_than_df["特征"] = iv_than_cols
        iv_than_df["删除原因"] = f"iv值大于等于{self.iv_upper}"

        select = select[np.setdiff1d(select.columns, iv_than_cols).tolist()]
        # 整理进一个df
        del_cols_df = pd.concat(
            [char_cols_df, one_value_df, more_null_df, iv_than_df, del_from_toad],
            axis=0,
            ignore_index=True,
        )
        del_cols_df = del_cols_df.drop_duplicates(
            subset=["特征"], keep="first", ignore_index=True
        ).set_index("特征")
        del_cols_df.index.name = None
        select_title = "训练集iv值" if self.train_test_split else "数据集iv值"
        select_cols_info = f"<br><br>对{select_title}大于等于{self.iv_threshold}，相关性小于等于{self.corr_threshold}进行入模特征选择。共筛选出{select.shape[1]-1}个入模特征。<br>"
        if str(self.iv_upper).isdigit():
            select_cols_info = f"<br><br>对{select_title}大于等于{self.iv_threshold}且小于{self.iv_upper}，相关性小于等于{self.corr_threshold}进行入模特征选择。共筛选出{select.shape[1]-1}个入模特征。<br>"

        # 4.1 入模特征iv值
        # 入模特征iv
        if self.show_params["model_fea"][0]:
            sub_params = self.show_params["model_fea"][1]
            sub_title_num = 1
            if sub_params["iv"] and select.shape[1] >= 2:
                select_iv = pd.DataFrame(
                    toad.quality(select, target=self.target_col, iv_only=True)["iv"]
                )
                select_iv["iv"] = select_iv["iv"].map(lambda x: "%.3f" % x)
                select_cols_info = (
                    select_cols_info
                    + f"<br><p>{h2_title_num}.1 入模特征IV值</p>"
                    + re.sub(
                        r"<table .*?>",
                        "<table border=1>",
                        select_iv.style.background_gradient(
                            subset=["iv"], cmap="RdYlGn"
                        ).to_html(),
                    )
                    + "<br>"
                )
                sub_title_num += 1

            # 4.2 入模特征相关性热力图
            if sub_params["corr"]:
                if select.shape[1] >= 2:
                    buff = Eda_Func.sns_corr(
                        select, sns_vars={"annot": True, "fmt": ".2f"}, is_pdf=True
                    )
                    html_img2 = fig_to_html(
                        buff, f"{h2_title_num}.{sub_title_num} 入模特征相关性热力图"
                    )
                    select_cols_info = select_cols_info + html_img2
                    sub_title_num += 1

            # 4.3 被删除特征统计html
            if sub_params["del"]:
                select_cols_info = (
                    select_cols_info
                    + f"<br><p>{h2_title_num}.{sub_title_num} 被删除特征原因统计</p>"
                    + del_cols_df.to_html()
                )

            body_list.append(select_cols_info)
            body_h2_list.append(f"{title_num_map[h2_title_num]}、入模特征筛选")
            h2_title_num += 1

        ################################ 五、模型效果展示#####################################
        # XGBoost模型
        if self.show_params["model_res"][0]:
            x_cols = [col for col in select.columns if col != self.target_col]
            self.select_feas = x_cols
            xgb_params = self.xgb_params
            if len(x_cols) >= 1:
                sub_params = self.show_params["model_res"][1]
                if sub_params["optuna"]:
                    optuna_result = Eda_Func.fast_ml_model(
                        data_dict["train"][x_cols + [self.target_col]].fillna(
                            self.fillna_value
                        ),
                        target=self.target_col,
                        model_params=xgb_params,
                        model_typ="xgb",
                        is_split=False,
                        is_optuna=True,
                        n_trials=sub_params["n_trials"],
                        show_plot=False,
                    )
                    xgb_params = optuna_result["model_params"]
                    xgb_params.update({"eval_metric": "auc"})
                # if not self.train_test_split:
                xgb_params.pop("early_stopping_rounds", None)
                xgb = XGBClassifier(**xgb_params)
                print("\n")
                msg_string = "开始训练XGBoost模型"
                print(f"{msg_string:=^62}")
                xgb.fit(
                    data_dict["train"][x_cols].fillna(self.fillna_value),
                    data_dict["train"].target,
                )
                data_name = ["train"]
                model_metrics = Eval_Module.model_metrics(
                    xgb,
                    data_dict["train"][x_cols].fillna(self.fillna_value),
                    data_dict["train"].target,
                )
                if (self.data_mode == 0 and self.train_test_split) or (
                    self.data_mode == 1
                ):
                    for idx_name, idx_data in data_dict.items():
                        if idx_name != "train":
                            data_name.append(idx_name)
                            model_metrics = pd.concat(
                                [
                                    model_metrics,
                                    Eval_Module.model_metrics(
                                        xgb,
                                        idx_data[x_cols].fillna(self.fillna_value),
                                        idx_data.target,
                                    ),
                                ],
                                axis=0,
                            )
                    model_metrics.reset_index(drop=True, inplace=True)
                    model_metrics["数据集"] = data_name
                self.model_metrics = model_metrics
                self.model_params = pd.DataFrame([xgb_params]).T
                self.model_params.columns = ["参数值"]
                print("\n")
                msg_string = "XGBoost相关性能指标如下"
                print(f"{msg_string:=^62}")
                print(self.model_metrics)
                model_bins = {}
                for idx_name, idx_data in data_dict.items():
                    cur_bins = Eval_Module.score_metric_ks(
                        xgb.predict_proba(idx_data[x_cols].fillna(self.fillna_value))[
                            :, 1
                        ],
                        idx_data[self.target_col],
                        n=sub_params["split_num"],
                        set_weight=None,
                        cut_method="freq",
                        is_change_weight=False,
                        asd=1,
                    ).round(2)
                    cur_bins.index = range(1, sub_params["split_num"] + 1)
                    model_bins[idx_name] = cur_bins
                # ROC曲线
                # Eval_Module.roc_plot(xgb, train[x_cols],train[self.target_col], test[x_cols],test[self.target_col], legend_loc='best')

                # 整理需要写入HTML的内容
                # 1.xgb模型参数
                # pa = pd.DataFrame()
                # pa.index = xgb_params.keys()
                # pa["参数值"] = xgb_params.values()
                # # border = """<hr style="FILTER: alpha(opacity=100,finishopacity=0,style=2)" color=#987cb9 SIZE=8>""" + "<br>"*2
                xgb_html = (
                    f"<p>{h2_title_num}.1 XGBoost模型参数</p>"
                    + self.model_params.to_html()
                    + "<br>"
                )

                # 2.模型指标效果
                xgb_html = (
                    xgb_html
                    + f"{h2_title_num}.2 XGBoost模型性能<br>"
                    + self.model_metrics.to_html()
                    + "<br>"
                )
                # 3.训练集100得分
                xgb_html = (
                    xgb_html
                    + f"{h2_title_num}.3 XGBoost模型【训练集】{sub_params['split_num']}等分<br>"
                    + re.sub(
                        r"<table .*?>",
                        "<table border=1>",
                        model_bins["train"]
                        .astype(str)
                        .style.background_gradient(
                            subset=["召回率", "精准率", "提升度"], cmap="RdYlGn"
                        )
                        .to_html(),
                    )
                    + "<br>"
                )
                # 3.1 测试集100等分
                module_six_subtitle = 4
                if (self.data_mode == 0 and self.train_test_split) or (
                    self.data_mode == 1
                ):
                    for idx_name, idx_bins in model_bins.items():
                        if idx_name != "train":
                            xgb_html = (
                                "<br>"
                                + xgb_html
                                + f"{h2_title_num}.{module_six_subtitle} XGBoost模型【{idx_name}】{sub_params['split_num']}等分<br>"
                                + re.sub(
                                    r"<table .*?>",
                                    "<table border=1>",
                                    idx_bins.astype(str)
                                    .style.background_gradient(
                                        subset=["召回率", "精准率", "提升度"], cmap="RdYlGn"
                                    )
                                    .to_html(),
                                )
                                + "<br>"
                            )
                            module_six_subtitle += 1

                # 4.特征重要性
                fi_imp = xgb.get_booster().get_score(importance_type="gain")
                if len(fi_imp) <= 20:
                    buff = BytesIO()
                    fig, ax = plt.subplots(figsize=(20, 15))
                    fi_viz = FeatureImportances(xgb, ax=ax, relative=True)
                    fi_viz.fit(
                        data_dict["train"][x_cols].fillna(self.fillna_value),
                        data_dict["train"][self.target_col],
                    )
                    fi_viz.show(outpath=buff, clear_figure=True)
                    xgb_html = xgb_html + fig_to_html(
                        buff,
                        fig_title=f"{h2_title_num}.{module_six_subtitle} XGBoost 模型特征重要性",
                    )
                else:
                    # 以DataFrame形式展示
                    fi_df = pd.DataFrame()
                    fi_df.index = fi_imp.keys()
                    fi_df["特征重要性"] = fi_imp.values()
                    fi_df["特征重要性"] = fi_df["特征重要性"].round(2)
                    fi_df = fi_df.sort_values(by="特征重要性", ascending=False)
                    xgb_html = (
                        xgb_html
                        + f"{h2_title_num}.{module_six_subtitle} XGBoost 模型特征重要性"
                        + "<br>"
                        + fi_df.to_html()
                    )
                module_six_subtitle += 1
                # 5.roc
                xgb_html = (
                    xgb_html
                    + "<br>"
                    + fig_to_html(
                        Eval_Module.roc_plot(
                            xgb,
                            {
                                "train": data_dict["train"][
                                    x_cols + [self.target_col]
                                ].fillna(self.fillna_value),
                            }
                            if (self.data_mode == 0 and not self.train_test_split)
                            else {
                                key: data_dict[key][x_cols + [self.target_col]].fillna(
                                    self.fillna_value
                                )
                                for key in data_dict.keys()
                            },
                            fig_size=(12, 9),
                            legend_loc="best",
                            is_pdf=True,
                        ),
                        fig_title=f"{h2_title_num}.{module_six_subtitle} XGBoost 模型ROC",
                    )
                )
                module_six_subtitle += 1

                # 6.ks
                xgb_html = (
                    xgb_html
                    + "<br>"
                    + fig_to_html(
                        Eval_Module.ks_plot(
                            {"train": model_bins["train"]}
                            if (self.data_mode == 0 and not self.train_test_split)
                            else model_bins,
                            fig_size=(12, 9),
                            is_pdf=True,
                        ),
                        fig_title=f"{h2_title_num}.7 XGBoost 模型KS",
                    )
                )
                module_six_subtitle += 1

                # 7.lift
                xgb_html = (
                    xgb_html
                    + "<br>"
                    + fig_to_html(
                        Eval_Module.lift_plot(
                            target_rate,
                            {"train": model_bins["train"]}
                            if (self.data_mode == 0 and not self.train_test_split)
                            else model_bins,
                            fig_size=(12, 9),
                            is_pdf=True,
                        ),
                        fig_title=f"{h2_title_num}.{module_six_subtitle} XGBoost模型Lift",
                    )
                )
                xgb_html = xgb_html + "<br>"
                # lift信息统计表
                lift_table = model_bins["train"][["分数段", "提升度"]].rename(
                    columns={"提升度": "train-lift"}
                )
                if (
                    self.data_mode == 0 and self.train_test_split
                ) or self.data_mode == 1:
                    for idx_name, idx_bins in model_bins.items():
                        if idx_name != "train":
                            lift_table = lift_table.merge(
                                idx_bins[["分数段", "提升度"]].rename(
                                    columns={"提升度": f"{idx_name}-lift"}
                                ),
                                how="left",
                                on=["分数段"],
                            )
                lift_table.insert(0, "bin index", range(1, sub_params["split_num"] + 1))
                lift_table.set_index("bin index", drop=True, inplace=True)
                xgb_html = xgb_html + lift_table.T.to_html() + "<br>" * 2
            else:
                xgb_html = "<p>没有筛选到相应的入模特征，放弃XGBoost模型的训练！</p>"

            body_list.append(xgb_html)
            body_h2_list.append(f"{title_num_map[h2_title_num]}、模型效果展示")
            h2_title_num += 1

        self.body_list = body_list
        self.body_h2_list = body_h2_list

        if self.save_name is not None and len(self.save_name) > 0:
            if len(self.id_col) >= 1:
                to_csv_cols = [self.id_col] + self.select_feas + [self.target_col]
                data[self.id_col] = id_series

            else:
                to_csv_cols = x_cols + [self.target_col]

            save_name = (
                self.save_name
                if self.save_name.endswith(".csv")
                else self.save_name + ".csv"
            )
            data[to_csv_cols].to_csv(save_name, encoding="utf-8", index=False)

    def transform(self):
        import time

        def transform_html(
            body_list, body_h2_list, header, title_name, report_name="EDA_REPORT"
        ):
            from io import BytesIO
            from lxml import etree
            import base64

            #             root = f"<h1 style='background:#336699; border:2;line-height:100px; padding-top: 2%; font-size:500%; font-weight: bold; color:#c1502e;text-align:center;'>{title_name}</center></h1> "+"<br>"*2
            root = (
                "<br>" * 2
                + f"<h1 id='C0' style='background-color:lightgray;font-family:newtimeroman;\
                    font-size:400%;text-align:center;border-radius: 15px 50px;color:#c1502e;'>{title_name}</center></h1> "
                + "<br>"
            )
            for idx, body_info in enumerate(body_list):
                idx_h2 = (
                    f"<h2 id='C{str(idx+1)}' style='background-color:#FFCC66; font-size:200%; font-weight: bold;text-align:center; color:#393E46;width:500px;\
                         border-radius: 15px 15px;'>{body_h2_list[idx]}</left></h2> "
                    + "<p></p>"
                )
                root = root + idx_h2 + body_info + "<br>" * 2
            #             print(root[:10000])
            root = header + root + "</body></html>"
            html = etree.HTML(root)
            tree = etree.ElementTree(html)
            tree.write(str(report_name) + ".html")

        # 将所有数据写为HTML
        # html头部设计
        html_header = "<html>\
<head>\
	<meta charset='UTF-8'>\
	<style>\
		.navbar {\
			position: fixed;\
			top: 0;\
			width: 100%;\
			background-color: #587498;\
			z-index: 1;\
		}\
		.navbar img {\
			float: left;\
			height: 50px;\
			padding: 10px;\
		}\
		.navbar ul {\
			float: right;\
			list-style-type: none;\
			margin: 0;\
			padding: 0;\
			display: flex;\
			align-items: center;\
		}\
		.navbar li {\
			padding: 10px;\
			margin-right: 10px;\
		}\
		.navbar a {\
			text-decoration: none;\
			color: #fff; \
			font-weight: bold;\
			font-size: 16px;\
		}\
	</style>\
</head>\
<body>\
	<div class='navbar'>\
		<img src='data:image/png;base64,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'>\
		<ul>\
			<li><a href='#C0'>首页</a></li>"
        for i, h2_title in enumerate(self.body_h2_list):
            cur_str = (
                "<li><a href=" + f"'#C{i+1}'>" + h2_title.split("、")[1] + "</a></li>"
            )
            html_header += cur_str
        html_header += "</ul></div>"

        transform_html(
            self.body_list,
            self.body_h2_list,
            html_header,
            self.report_title,
            report_name=self.report_file_name,
        )
        self.end_time = time.time()
        m, s = divmod(time.time() - self.start_time, 60)
        print(f"报告已生成，报告名称为：{self.report_file_name}.html")
        print(f"脚本运行耗时：{int(m)}分钟{int(s)}秒！")
