现在的problem集合的问题说明

## 1. 现在有的problem

- 力学问题
    - 重力问题
    - 浮力问题
    - 挤压形变
    - 碰撞和冲击
- 热学问题
    - 物体融化 冰雪
    - 液体蒸发
    - 物体升华 干冰
    - 状态迁移 固液气
- 材料学问题
    - 颜色混合问题
    - 溶解过程
    - 材料切割折叠断裂
    - 燃烧的材料是否合理

## 2. 测试结果

使用模型 Gemini 2.5 flash

video 1_A到5_B 是从vmem github page获取的视频，其中A是不一致性糟糕的视频，而B是不一致性良好的视频
video 6开始 是从phyvideo2 github page获取的视频，均为存在不规律物理现象的视频

- 1_A.mp4 : 模型将视频前后的不一致性判定为走廊的柱子发生了不规律形变，回答了no，个人认为这还算合理
- 1_B.mp4 : 模型在这里做出了和1_A一致的回答，同样认为柱子发生了不规律形变，一些细微的扭动
- 4_A.mp4 : 模型将前后的不一致也判定为不规律形变和断裂，同时前后水景的不一致，也认为发生不正常的蒸发现象
- 4_B.mp4 : 这里模型做出了和4_B不同的回答，没有认为发生形变，而是一处石柱的颜色不太自然，让模型认为这是冰块，认定为这个温度下不该出现冰块，判定为不自然
- 6.mp4 : 无法测评出这个石头滚动是在无视重力向上滚动，能够判断这个石头撞击产生灰尘是正常的
- 7.mp4 : 能够正确评测船的漂浮是正确的物理现象，但是无法观测到船桨不自然的形变

## 3. problem details

```python

ECHANICS_QUESTIONS: List[PhysicsQuestion] = [
    PhysicsQuestion(
        qid="gravity",
        dimension="mechanics",
        question="Do free-moving objects downward consistently with gravity?",
        success_condition="First judge whether there is object go up or down, Objects (e.g., balls, cans) should move downward instead of upward unless supported.",
    ),
    PhysicsQuestion(
        qid="buoyancy",
        dimension="mechanics",
        question="Do objects on or in a fluid behave consistently with buoyancy (floating items stay near the surface, sinking items submerge)?",
        success_condition="Floating objects should remain on/near the surface; dense objects should descend.",
    ),
    PhysicsQuestion(
        qid="compression",
        dimension="mechanics",
        question="When solids are squeezed or stressed, do they visibly deform in a plausible manner?",
        success_condition="E.g., cans dent when crushed; soft materials compress smoothly under load; A wooden stick without force will not suddenly deform..",
    ),
    PhysicsQuestion(
        qid="impact",
        dimension="mechanics",
        question="After collisions or impacts, do objects transition to a reasonable post-impact state?",
        success_condition="Look for momentum transfer, bouncing, shattering, or resting poses that match the impact.",
    ),
]


# 4-b: Thermotics (热学行为正确性)
# 4-b: Thermotics (热学行为正确性 - 六种相变)
THERMOTICS_QUESTIONS: List[PhysicsQuestion] = [
    PhysicsQuestion(
        qid="melting",
        dimension="thermotics",
        question="When a solid is heated, does it show signs of melting (solid → liquid transition)?",
        success_condition="Solids should gradually transition to liquid when heated above their melting point (e.g., ice to water, wax melting).",
    ),
    PhysicsQuestion(
        qid="sublimation",
        dimension="thermotics",
        question="Does the solid directly turn into gas without passing through the liquid phase (solid → gas transition)?",
        success_condition="Certain solids should directly transition to gas when heated (e.g., dry ice to CO2 gas, iodine to purple vapor).",
    ),
    PhysicsQuestion(
        qid="vaporization",
        dimension="thermotics",
        question="When a liquid is heated, does it show signs of vaporization (liquid → gas transition, evaporation or boiling)?",
        success_condition="Liquids should transition to gas when heated (e.g., water evaporating, alcohol boiling) or exposed to air over time.",
    ),
    PhysicsQuestion(
        qid="condensation",
        dimension="thermotics",
        question="When a gas is cooled, does it show signs of condensation (gas → liquid transition)?",
        success_condition="Gas should transition to liquid when cooled below its condensation point (e.g., steam to water droplets, breath fog in cold air).",
    ),
    PhysicsQuestion(
        qid="deposition",
        dimension="thermotics",
        question="Does the gas directly turn into solid without passing through the liquid phase (gas → solid transition)?",
        success_condition="Certain gases should directly transition to solid when cooled (e.g., water vapor to frost, CO2 gas to dry ice).",
    ),
    PhysicsQuestion(
        qid="freezing",
        dimension="thermotics",
        question="When a liquid is cooled, does it show signs of freezing (liquid → solid transition)?",
        success_condition="Liquids should transition to solid when cooled below their freezing point (e.g., water to ice, wax solidifying).",
    ),
]


# 4-c: Material (材料属性遵从性)
MATERIAL_QUESTIONS: List[PhysicsQuestion] = [
    PhysicsQuestion(
        qid="color_mixing",
        dimension="material",
        question="When different colored liquids or paints mix, do they produce the correct resulting color?",
        success_condition="Red + Yellow → Orange, Blue + Yellow → Green, Red + Blue → Purple, etc.",
    ),
    PhysicsQuestion(
        qid="solubility",
        dimension="material",
        question="Do soluble materials (sugar, salt) dissolve properly when placed in water or other solvents?",
        success_condition="Soluble substances should gradually disperse and become invisible/transparent in the solvent.",
    ),
    PhysicsQuestion(
        qid="hardness",
        dimension="material",
        question="Do materials with different hardness levels behave correctly when cut, folded, or broken?",
        success_condition="Soft materials (paper, cloth) should fold/tear easily; hard materials (metal, stone) should resist or break sharply.",
    ),
    PhysicsQuestion(
        qid="combustibility",
        dimension="material",
        question="Do flammable materials burn correctly, producing fire, smoke, or char?",
        success_condition="Wood, paper, fabric should ignite and produce flames/smoke; non-flammable materials should not.",
    ),
]


```