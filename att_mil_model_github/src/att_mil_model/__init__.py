"""att_mil_model: Attention-based Multiple Instance Learning (MIL) utilities.

该包包含：
- 注意力 MIL 模型（分类/回归）
- Bag 级别数据集读取
- 交叉验证训练、早停
- Optuna 超参数搜索示例

说明：为了便于在 GitHub 发布，本仓库避免硬编码真实路径；
请通过命令行参数/配置文件传入数据与输出目录。
"""

from .models.att_mil_model import att_mil_model, AttentionModule

__all__ = ["att_mil_model", "AttentionModule"]
