# dataset/aoss_imagenet.py
import torch
from torch.utils.data import Dataset
from PIL import Image
import boto3
import botocore
from botocore.config import Config
import io
import sys
import os


class AOSSImageNetDataset(Dataset):
    def __init__(self, bucket: str, transform=None, filelist_path='./imagenet_train_filelist.txt'):
        """
        AOSS ImageNet Dataset

        Args:
            filelist_path: 文件列表路径，每行一个对象key
            bucket: AOSS bucket名称
            transform: torchvision transforms
        """
        # 清除代理设置
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['NO_PROXY'] = '*'

        # 读取文件列表
        with open(filelist_path) as f:
            self.filelist = f.read().splitlines()

        self.transform = transform
        self.bucket = bucket

        # AOSS配置
        # self.aws_access_key_id = '0198A1B9771F7BAAA9A55AC5B51ACC2F'
        self.aws_access_key_id = '01997084CBF777519D5F10EC029154C6'
        # self.aws_secret_access_key = '0198A1B9771F7B9D998F202B044BE13C'
        self.aws_secret_access_key = '01997084CBF7774488E90D66F4FCB83D'
        # self.endpoint_url = 'http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn' # 内网地址
        self.endpoint_url = 'http://aoss.cn-sh-01b.sensecoreapi-oss.cn' # 外网地址

        self.s3_client = None
        self.config = Config(
            connect_timeout=300,
            read_timeout=300,
            retries={'max_attempts': 20, 'mode': 'adaptive'}
        )

    def _ensure_s3_client(self):
        """延迟初始化S3客户端"""
        if self.s3_client is None:
            try:
                self.s3_client = boto3.client(
                    's3',
                    endpoint_url=self.endpoint_url,
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    config=self.config
                )
            except Exception as e:
                print(f"❌ 连接AOSS失败: {e}")
                sys.exit(1)

    def load_image_from_aoss(self, object_key):
        """从AOSS加载图像"""
        self._ensure_s3_client()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.s3_client.get_object(Bucket=self.bucket, Key=object_key)
                img_bytes = response['Body'].read()
                img_buffer = io.BytesIO(img_bytes)
                img = Image.open(img_buffer).convert('RGB')
                return img
            except botocore.exceptions.ReadTimeoutError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ 读取超时，重试 {attempt + 1}/{max_retries}: {object_key}")
                    continue
                print(f"❌ 读取超时 ({object_key}): {e}")
                return None
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    print(f"⚠️ 文件不存在: {object_key}")
                else:
                    print(f"❌ 客户端错误 ({object_key}): {e}")
                return None
            except Exception as e:
                print(f"❌ 加载图像出错 ({object_key}): {e}")
                return None

        return None

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index: int):
        img_path = self.filelist[index]
        img = self.load_image_from_aoss(img_path)

        if img is None:
            # 如果加载失败，返回黑色图像（或者可以跳过）
            print(f"⚠️ 使用占位图像替代: {img_path}")
            img = Image.new('RGB', (256, 256), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        # 返回图像和dummy label（VQ训练不需要label）
        return img, 0


def build_aoss_imagenet(args, transform):
    """构建AOSS ImageNet数据集"""
    return AOSSImageNetDataset(
        filelist_path=args.data_path,  # 这里data_path是filelist路径
        bucket=args.aoss_bucket,
        transform=transform
    )
