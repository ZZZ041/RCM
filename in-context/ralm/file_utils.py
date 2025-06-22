import logging
import os

logger = logging.getLogger()    # 创建一个全局日志记录器
logger.setLevel(logging.INFO)   # 设置日志级别为INFO
log_formatter = logging.Formatter("[%(thread)s] %(asctime)s [%(levelname)s] %(name)s: %(message)s")     # 创建一个日志格式化器
console = logging.StreamHandler()   # 创建一个日志处理器，将日志信息输出到控制台
console.setFormatter(log_formatter)     # 为日志处理器设置格式化器
logger.addHandler(console)  # 将日志处理器添加到日志记录器


# 打印和保存传入的参数配置
def print_args(args, output_dir=None, output_file=None):
    assert output_dir is None or output_file is None

    logger.info(" **************** CONFIGURATION **************** ")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (30 - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info(" **************** CONFIGURATION **************** ")

    if output_dir is not None or output_file is not None:
        output_file = output_file or os.path.join(output_dir, "args.txt")
        with open(output_file, "w") as f:
            for key, val in sorted(vars(args).items()):
                keystr = "{}".format(key) + (" " * (30 - len(key)))
                f.write(f"{keystr}   {val}\n")
