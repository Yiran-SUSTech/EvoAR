import boto3
import botocore
import sys
import os

# --- 1. AOSS 配置 (基于您提供的 AOSSImageNetDataset 类) ---

# 注意：在生产环境中，硬编码密钥是不安全的。
# 建议使用环境变量、配置文件或 IAM 角色。
AWS_ACCESS_KEY_ID = '0198A1B9771F7BAAA9A55AC5B51ACC2F'
AWS_SECRET_ACCESS_KEY = '0198A1B9771F7B9D998F202B044BE13C'
ENDPOINT_URL = 'http://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn'

# --- 2. 待配置的参数 ---

# 替换为您的 AOSS 存储桶名称
BUCKET_NAME = 'imagenet' 
# 假设 ImageNet 训练数据在桶中的前缀是 'train/'
# 示例：文件路径为 train/n01440764/n01440764_10026.JPEG
PREFIX = 'train/' 

# 生成的文件列表名称
OUTPUT_FILE = 'imagenet_train_filelist.txt'

# --- 3. 核心功能函数 ---

def generate_aoss_filelist():
    """
    连接 AOSS 并遍历指定前缀下的所有对象键，生成文件列表。
    """
    print(f"尝试连接到 AOSS: {ENDPOINT_URL}")
    print(f"存储桶: {BUCKET_NAME}, 前缀: {PREFIX}")
    
    # 清除代理设置 (与您的数据集类保持一致)
    os.environ['HTTP_PROXY'] = ''
    os.environ['HTTPS_PROXY'] = ''
    os.environ['NO_PROXY'] = '*'

    try:
        # 创建 S3 客户端
        s3_client = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            # 使用您代码中的连接配置
            config=botocore.config.Config(
                connect_timeout=300,
                read_timeout=300,
                retries={'max_attempts': 20, 'mode': 'adaptive'}
            )
        )
    except Exception as e:
        print(f"❌ 初始化S3客户端失败: {e}")
        sys.exit(1)

    filelist = []
    object_count = 0
    
    # 使用 Paginator 处理大量对象，自动处理 NextContinuationToken
    paginator = s3_client.get_paginator('list_objects_v2')
    
    try:
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX)

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    # 仅添加 JPEG 文件（ImageNet 标准格式），并忽略可能是目录的对象（通常以斜杠结尾）
                    if key.endswith(('.JPEG', '.jpeg', '.jpg', '.png')) and not key.endswith('/'):
                        filelist.append(key)
                        object_count += 1
                        if object_count % 10000 == 0:
                            print(f"已发现 {object_count} 个文件...")
            
    except botocore.exceptions.ClientError as e:
        print(f"❌ 客户端错误，请检查存储桶名称和权限: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 遍历AOSS对象时发生错误: {e}")
        sys.exit(1)

    # 将文件列表写入本地文件
    if filelist:
        try:
            with open(OUTPUT_FILE, 'w') as f:
                f.write('\n'.join(filelist))
            
            print("-" * 50)
            print(f"✅ 文件列表生成成功！")
            print(f"总计发现 {object_count} 个文件。")
            print(f"列表已保存到: {OUTPUT_FILE}")
            print("-" * 50)
        except IOError as e:
            print(f"❌ 写入本地文件失败: {e}")
            sys.exit(1)
    else:
        print("⚠️ 在指定前缀下未发现任何文件。请检查 BUCKET_NAME 和 PREFIX 是否正确。")


if __name__ == '__main__':
    # **重要：请先将 BUCKET_NAME 替换为您的真实存储桶名称！**
    if BUCKET_NAME == 'your-imagenet-bucket-name':
        print("请先修改脚本中的 BUCKET_NAME 变量为您的 AOSS 存储桶名称，然后再次运行。")
        sys.exit(1)
        
    generate_aoss_filelist()