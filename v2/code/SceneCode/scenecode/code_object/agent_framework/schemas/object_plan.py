"""
ObjectPlan Schema Definition
定义物体规划的数据结构
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class ObjectCategory(Enum):
    """物体类别枚举"""
    CHAIR = "chair"
    TABLE = "table"
    STOOL = "stool"
    DESK = "desk"
    BENCH = "bench"
    CABINET = "cabinet"
    SHELF = "shelf"


class ShapeType(Enum):
    """基础形状类型"""
    CUBE = "cube"
    CYLINDER = "cylinder"
    SPHERE = "sphere"
    CONE = "cone"
    TORUS = "torus"
    CUSTOM = "custom"


class RelativePosition(Enum):
    """相对位置关系"""
    ON_TOP = "on_top"           # 在...上方
    BELOW = "below"             # 在...下方
    BEHIND = "behind"           # 在...后方
    IN_FRONT = "in_front"       # 在...前方
    LEFT_OF = "left_of"         # 在...左侧
    RIGHT_OF = "right_of"       # 在...右侧
    ATTACHED_TO = "attached_to" # 连接到...
    INSIDE = "inside"           # 在...内部
    SURROUNDING = "surrounding" # 环绕...


@dataclass
class Dimensions:
    """尺寸信息"""
    width: float   # X 轴
    depth: float   # Y 轴
    height: float  # Z 轴
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "width": self.width,
            "depth": self.depth,
            "height": self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Dimensions":
        return cls(
            width=data.get("width", 1.0),
            depth=data.get("depth", 1.0),
            height=data.get("height", 1.0)
        )


@dataclass
class Position:
    """位置信息"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Position":
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0)
        )


@dataclass
class Rotation:
    """旋转信息（欧拉角，单位：度）"""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "Rotation":
        return cls(
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            z=data.get("z", 0.0)
        )


@dataclass
class PartRelation:
    """部件关系"""
    target_part: str                    # 目标部件名称
    relation: RelativePosition          # 相对位置关系
    offset: Optional[Position] = None   # 额外偏移
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_part": self.target_part,
            "relation": self.relation.value,
            "offset": self.offset.to_dict() if self.offset else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartRelation":
        return cls(
            target_part=data["target_part"],
            relation=RelativePosition(data["relation"]),
            offset=Position.from_dict(data["offset"]) if data.get("offset") else None
        )


@dataclass
class PartShape:
    """部件形状描述"""
    base_shape: ShapeType               # 基础形状
    dimensions: Dimensions              # 尺寸
    modifiers: List[str] = field(default_factory=list)  # 修改器列表（如 bevel, smooth）
    description: str = ""               # 形状的文字描述
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_shape": self.base_shape.value,
            "dimensions": self.dimensions.to_dict(),
            "modifiers": self.modifiers,
            "description": self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartShape":
        base_shape_raw = data.get("base_shape", "cube")
        if isinstance(base_shape_raw, ShapeType):
            base_shape = base_shape_raw
        else:
            shape_str = str(base_shape_raw).strip().lower()
            try:
                base_shape = ShapeType(shape_str)
            except ValueError:
                # Fallback for any non-enum shape token from LLM (e.g. "curve")
                base_shape = ShapeType.CUSTOM

        return cls(
            base_shape=base_shape,
            dimensions=Dimensions.from_dict(data.get("dimensions", {})),
            modifiers=data.get("modifiers", []),
            description=data.get("description", "")
        )


@dataclass
class PartPlan:
    """单个部件的规划"""
    name: str                           # 部件名称（如 seat, leg_01）
    part_type: str                      # 部件类型（如 seat, leg, backrest）
    shape: PartShape                    # 形状描述
    position: Position                  # 位置（世界坐标）
    rotation: Rotation                  # 旋转
    is_symmetric: bool = False          # 是否有对称部件
    symmetric_axis: str = "x"           # 对称轴
    is_movable: bool = False            # 是否可动（如抽屉/柜门/滑门）
    must_be_independent: bool = False   # 是否必须独立对象（不可用阵列/镜像合并表示）
    description: str = ""               # 部件的文字描述
    priority: int = 0                   # 构建优先级（数字越小越先构建）
    material: Dict[str, Any] = field(default_factory=dict)  # 材质/纹理信息
    sub_parts: List["PartPlan"] = field(default_factory=list)  # 内部子部件（仅一层，不可再嵌套）
    instances: List[Dict[str, Any]] = field(default_factory=list)  # 同构实例列表（仅位置/旋转不同的独立副本）
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "part_type": self.part_type,
            "shape": self.shape.to_dict(),
            "position": self.position.to_dict(),
            "rotation": self.rotation.to_dict(),
            "is_symmetric": self.is_symmetric,
            "symmetric_axis": self.symmetric_axis,
            "is_movable": self.is_movable,
            "must_be_independent": self.must_be_independent,
            "description": self.description,
            "priority": self.priority,
            "material": self.material
        }
        if self.sub_parts:
            result["sub_parts"] = [sp.to_dict() for sp in self.sub_parts]
        if self.instances:
            result["instances"] = self.instances
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartPlan":
        # 解析 sub_parts（仅一层，子部件不再递归解析 sub_parts）
        sub_parts_data = data.get("sub_parts", [])
        sub_parts = []
        for sp_data in sub_parts_data:
            sp_data.pop("sub_parts", None)  # 强制去除嵌套，仅支持一层
            sub_parts.append(cls.from_dict(sp_data))
        
        return cls(
            name=data["name"],
            part_type=data.get("part_type", data["name"]),
            shape=PartShape.from_dict(data.get("shape", {})),
            position=Position.from_dict(data.get("position", {})),
            rotation=Rotation.from_dict(data.get("rotation", {})),
            # parent=data.get("parent"),
            # relations=[PartRelation.from_dict(r) for r in data.get("relations", [])],
            is_symmetric=data.get("is_symmetric", False),
            symmetric_axis=data.get("symmetric_axis", "x"),
            is_movable=data.get("is_movable", False),
            must_be_independent=data.get("must_be_independent", False),
            description=data.get("description", ""),
            priority=data.get("priority", 0),
            material=data.get("material", {}),
            sub_parts=sub_parts,
            instances=data.get("instances", [])
        )


@dataclass
class ObjectPlan:
    """完整的物体规划"""
    category: str            # 物体类别
    name: str                           # 物体名称（如 chair_001）
    description: str                    # 物体整体描述
    parts: List[PartPlan]               # 部件列表
    total_dimensions: Dimensions        # 整体尺寸估计
    style: str = ""                     # 风格描述（如 modern, classic）
    material_hints: Dict[str, str] = field(default_factory=dict)  # 材质提示
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "description": self.description,
            "parts": [p.to_dict() for p in self.parts],
            "total_dimensions": self.total_dimensions.to_dict(),
            "style": self.style,
            "material_hints": self.material_hints,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """导出为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, filepath: str) -> None:
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectPlan":
        return cls(
            category=data["category"],
            name=data["name"],
            description=data.get("description", ""),
            parts=[PartPlan.from_dict(p) for p in data.get("parts", [])],
            total_dimensions=Dimensions.from_dict(data.get("total_dimensions", {})),
            style=data.get("style", ""),
            material_hints=data.get("material_hints", {}),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "ObjectPlan":
        """从 JSON 字符串加载"""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def load(cls, filepath: str) -> "ObjectPlan":
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
    
    def get_part_by_name(self, name: str) -> Optional[PartPlan]:
        """根据名称获取部件"""
        for part in self.parts:
            if part.name == name:
                return part
        return None
    
    def get_parts_by_type(self, part_type: str) -> List[PartPlan]:
        """根据类型获取部件列表"""
        return [p for p in self.parts if p.part_type == part_type]
    
    def get_parts_sorted_by_priority(self) -> List[PartPlan]:
        """按优先级排序的部件列表"""
        return sorted(self.parts, key=lambda p: p.priority)


# ==================== Action Trace Schema ====================

@dataclass
class BlenderAction:
    """单个 Blender 操作"""
    action_type: str                    # 操作类型
    parameters: Dict[str, Any]          # 操作参数
    result: Optional[str] = None        # 操作结果
    timestamp: Optional[str] = None     # 时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "parameters": self.parameters,
            "result": self.result,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlenderAction":
        return cls(
            action_type=data["action_type"],
            parameters=data.get("parameters", {}),
            result=data.get("result"),
            timestamp=data.get("timestamp")
        )


@dataclass
class ActionTrace:
    """部件构建的操作轨迹"""
    part_name: str                      # 部件名称
    actions: List[BlenderAction]        # 操作列表
    final_code: str = ""                # 最终生成的代码
    iterations: int = 0                 # 迭代次数
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "part_name": self.part_name,
            "actions": [a.to_dict() for a in self.actions],
            "final_code": self.final_code,
            "iterations": self.iterations
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionTrace":
        return cls(
            part_name=data["part_name"],
            actions=[BlenderAction.from_dict(a) for a in data.get("actions", [])],
            final_code=data.get("final_code", ""),
            iterations=data.get("iterations", 0)
        )
    
    @classmethod
    def load(cls, filepath: str) -> "ActionTrace":
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_dict(json.loads(f.read()))
