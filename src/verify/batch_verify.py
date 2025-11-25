import json
import argparse
import sys
from tqdm import tqdm
from verify import BarrierValidator


class BatchTester:
    """
    用于批量验证 Barrier Certificate 数据集的测试器。
    """
    def __init__(self, file_path: str):
        if not file_path.endswith('.json') and not file_path.endswith('.jsonl'):
            raise ValueError("File must be a .json or .jsonl file.")
        self.file_path = file_path
        self.results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'errors': 0
        }

    def _load_data(self):
        """根据文件扩展名加载数据"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                if self.file_path.endswith('.jsonl'):
                    # .jsonl 文件，每行是一个独立的 JSON 对象
                    return [line.strip() for line in f if line.strip()]
                else:
                    # .json 文件，整个文件是一个 JSON 数组
                    data_list = json.load(f)
                    # 将列表中的每个dict转换回JSON字符串，以匹配输入格式
                    return [json.dumps(item) for item in data_list]
        except FileNotFoundError:
            print(f"❌ Error: File not found at '{self.file_path}'")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"❌ Error: Could not decode JSON from '{self.file_path}'. Please check the file format.")
            sys.exit(1)


    def run_tests(self, degs: dict):
        """
        执行批量测试。
        """
        print(f"🚀 Starting validation for: {self.file_path}")
        
        dataset_strings = self._load_data()
        self.results['total'] = len(dataset_strings)

        if self.results['total'] == 0:
            print("⚠️ Warning: The input file is empty or contains no data.")
            return

        # 使用tqdm创建进度条
        for i, dataset_string in enumerate(tqdm(dataset_strings, desc="Validating")):
            try:
                validator = BarrierValidator(dataset_string)
                is_valid = validator.verify_all(degs=degs)
                
                if is_valid:
                    self.results['success'] += 1
                else:
                    self.results['failed'] += 1

            except Exception as e:
                # 任何在初始化或验证期间的异常都算作错误
                self.results['errors'] += 1
                print(f"\n❗ Error processing item {i+1}: {e}")

    def print_report(self):
        """
        打印最终的统计报告。
        """
        total = self.results['total']
        success = self.results['success']
        failed = self.results['failed']
        errors = self.results['errors']

        print("\n" + "="*40)
        print("📊 Batch Validation Report")
        print("="*40)
        print(f"🔹 Total items processed: {total}")
        print(f"✅ Successful validations: {success}")
        print(f"❌ Failed validations: {failed}")
        print(f"❗ Errors during processing: {errors}")
        print("-" * 40)
        
        if total > 0:
            success_rate = (success / total) * 100
            print(f"📈 Success Rate: {success_rate:.2f}%")
        else:
            print("📈 Success Rate: N/A (no data processed)")
            
        print("="*40)


if __name__ == '__main__':
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="Batch validator for Barrier Certificates from a .json or .jsonl file."
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the input .json or .jsonl file."
    )
    args = parser.parse_args()

    # --- 定义验证参数 ---
    # SOS (Sum-of-Squares) 验证中多项式的次数
    validation_degrees = {
        'init': 2, 
        'unsafe': 2, 
        'lie_s': 2, 
        'lie_lambda': 2
    }

    # --- 执行测试 ---
    tester = BatchTester(file_path=args.filepath)
    tester.run_tests(degs=validation_degrees)
    tester.print_report()