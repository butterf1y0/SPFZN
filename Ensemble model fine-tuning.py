import os
import traceback

from pyabsa import AspectPolarityClassification as APC
from pyabsa import ModelSaveOption, DeviceTypeOption


def assert_dataset_files(dataset_dir: str):
    """必须同时存在 train.dat.apc / test.dat.apc / valid.dat.apc"""
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"[DATASET] dataset_dir 不存在或不是目录：{dataset_dir}")

    train_path = os.path.join(dataset_dir, "train.dat.apc")
    test_path = os.path.join(dataset_dir, "test.dat.apc")
    valid_path = os.path.join(dataset_dir, "valid.dat.apc")

    missing = []
    if not os.path.isfile(train_path):
        missing.append("train.dat.apc")
    if not os.path.isfile(test_path):
        missing.append("test.dat.apc")
    if not os.path.isfile(valid_path):
        missing.append("valid.dat.apc")

    if missing:
        raise ValueError(
            f"[DATASET] 在目录 {dataset_dir} 找不到必要文件：{', '.join(missing)}\n"
            f"必须同时包含：train.dat.apc、test.dat.apc、valid.dat.apc。训练已中断。"
        )

    print("[DATASET] 三个必要文件均已检测到。")
    print(f"[DATASET] train: {train_path}")
    print(f"[DATASET] test : {test_path}")
    print(f"[DATASET] valid: {valid_path}")


def build_config(local_bert_dir: str):

    config = APC.APCConfigManager.get_apc_config_chinese()

    config.spacy_model = "zh_core_web_sm"

    config.model = APC.APCModelList.FAST_LSA_T_V2

    config.pretrained_bert = local_bert_dir

    # 训练超参
    config.num_epoch = 16
    config.evaluate_begin = 1
    config.batch_size = 16
    config.max_seq_len = 128
    config.log_step = 100
    config.dropout = 0.3
    config.learning_rate = 1e-5
    config.l2reg = 1e-5
    config.cache_dataset = False
    config.use_amp = True
    config.seed = 52

    return config


def train_one_model(dataset_dir: str, local_bert_dir: str, save_subdir: str, model_name: str):
    os.makedirs(save_subdir, exist_ok=True)

    config = build_config(local_bert_dir=local_bert_dir)

    print("\n" + "=" * 90)
    print(f"[Ensemble] Start training model={model_name}")
    print(f"[Ensemble] dataset_dir: {dataset_dir}")
    print(f"[Ensemble] bert_dir   : {local_bert_dir}")
    print(f"[Ensemble] save_to    : {save_subdir}")
    print("=" * 90)

    _ = APC.APCTrainer(
        config=config,
        dataset=dataset_dir,
        auto_device=DeviceTypeOption.AUTO,
        checkpoint_save_mode=ModelSaveOption.SAVE_MODEL_STATE_DICT,
        path_to_save=save_subdir,
        load_aug=False,
        # from_checkpoint=r"..."
    )

    print(f"[Ensemble] Done model={model_name}. checkpoints saved to: {save_subdir}")


def main():

    # 数据集目录：必须包含 train.dat.apc、test.dat.apc、valid.dat.apc
    dataset_dir = r"./data/apc_dataset"

    # 训练结果保存目录
    save_root = r"./outputs/ensemble"

    os.makedirs(save_root, exist_ok=True)

    # 5个不同子模型
    bert_models = [
        {
            "name": "model_1",
            "path": r"./pretrained_models/model_1"
        },
        {
            "name": "model_2",
            "path": r"./pretrained_models/model_2"
        },
        {
            "name": "model_3",
            "path": r"./pretrained_models/model_3"
        },
        {
            "name": "model_4",
            "path": r"./pretrained_models/model_4"
        },
        {
            "name": "model_5",
            "path": r"./pretrained_models/model_5"
        },
    ]


    assert_dataset_files(dataset_dir)

    for model_info in bert_models:
        if not os.path.isdir(model_info["path"]):
            raise ValueError(f"[MODEL] 模型目录不存在：{model_info['path']}")

    failed = []
    for model_info in bert_models:
        model_name = model_info["name"]
        local_bert_dir = model_info["path"]
        subdir = os.path.join(save_root, model_name)

        try:
            train_one_model(
                dataset_dir=dataset_dir,
                local_bert_dir=local_bert_dir,
                save_subdir=subdir,
                model_name=model_name,
            )
        except Exception as e:
            failed.append((model_name, str(e)))
            print("\n" + "!" * 90)
            print(f"[Ensemble][FAILED] model={model_name} error={e}")
            print("Traceback:")
            print(traceback.format_exc())
            print("!" * 90)

    print("\n" + "=" * 90)
    print(f"[Ensemble] All done. save_root={save_root}")
    if failed:
        print("[Ensemble] Failed models:", failed)
    else:
        print("[Ensemble] All models trained successfully.")
    print("=" * 90)


if __name__ == "__main__":
    main()