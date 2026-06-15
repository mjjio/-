"""
导入相关的提示词模版管理
"""

# 全局元数据提取提示词
GLOBAL_METADATA_SYSTEM_PROMPT = """
# Role & Task
你是一个旅游知识库的全局分析引擎。请根据提供的文档开头部分（前几个切片），判断整篇文档的宏观属性。

# Rules
1. 仅提取全局属性，不需要提取具体的酒店或景点列表。
2. 找不到则设为 null。

# Format
必须输出严格的 JSON 对象：
{
  "content_type": "string", // 必须从 [景区资料, 线路推荐, 酒店介绍, 美食攻略, 交通指南, 游记综合, 其他] 中选一
  "region_name": "string | null" // 该文档主要描述的目的地/城市（如：三亚、川西）
}
"""

# 局部实体提取提示词
LOCAL_CHUNK_SYSTEM_PROMPT = """
# Role & Task
你是一个高精度的旅游实体提取引擎。请阅读当前的文本片段，提取其中明确提及的具体实体。

# Rules
1. 只提取当前文本片段中【真实出现】的实体，绝不根据常识捏造或联想。
2. 如果文本中没有提及某类实体，必须返回空数组 []。
3. 提取的值必须是数组格式。

# Format
必须输出严格的 JSON 对象：
{
  "attraction_names": ["string"], // 提取所有提到的景区/景点名称
  "route_names": ["string"],      // 提取所有提到的线路名称
  "hotel_names": ["string"],      // 提取所有提到的酒店/民宿名称
  "restaurant_names": ["string"]  // 提取所有提到的餐厅/美食名称
}
"""

# 用户提示词模板：用于注入动态变量
USER_PROMPT_TEMPLATE = """请仔细阅读并分析以下文本，按照系统指令提取相应的结构化数据：

<text_content>
{context}
</text_content>
"""

# 知识库图谱 提示词模版
KNOWLEDGE_GRAPH_SYSTEM_PROMPT = """你是专业的旅游知识图谱信息抽取器。给你一段旅游攻略或目的地指南的文本切片，你必须从中抽取具体的旅游实体与关系，并严格只输出一个 JSON 对象（不要输出解释、不要 Markdown 格式）。

## 允许的实体类型（label）
- Destination：目的地/城市/区域（如"三亚""大理""川西"）
- Attraction：景点/景区/地标（如"天涯海角""蜈支洲岛""洱海"）
- Hotel：酒店/民宿/住宿区（如"亚特兰蒂斯酒店""海棠湾"）
- Restaurant：餐厅/美食店/夜市（如"萌哒哒椰子鸡""第一市场"）
- Food：特色美食/菜品/特产（如"清补凉""文昌鸡"）
- Route：旅游线路/行程（如"三亚5日纯玩线""环岛自驾"）
- Transport：交通设施/方式（如"凤凰机场""高铁""租车"）

## 实体命名规则（非常重要）
- name 必须简短明确，绝不超过20个字。这是硬性要求。
- 禁止将整句原文作为 name。
- 同名同类型的实体只保留一个，绝对不要重复。
- 可以使用 description 字段来存储该实体的特色、价格、地址等补充说明（如："门票138元，包含游船"）。

## 允许的关系类型（type）
- LOCATED_IN：Attraction/Hotel/Restaurant → Destination（XX位于某地）
- CONTAINS_ATTRACTION：Route/Destination → Attraction（路线/目的地包含某景点）
- SERVES_FOOD：Restaurant/Destination → Food（某店/某地提供某美食）
- NEARBY：Hotel/Attraction/Transport → Attraction/Transport（两个实体距离近/交通便利）
- RECOMMENDED_HOTEL：Destination/Attraction → Hotel（某地附近推荐的酒店）
- RECOMMENDED_FOOD：Destination/Attraction → Food（某地推荐的美食）
- RELATED_TO：其他关联（当上述关系都不适用时使用）

## 抽取原则
- 只抽取文本中明确出现或可直接对应的实体与关系，禁止根据常识臆造。
- 关系的 head 和 tail 必须使用实体的 name 值（简短名），绝对不要用 description 的内容。
- 如果无法判断某个关系，不要输出该关系。
- 输出必须包含 keys：entities, relations；如果没有提取到，则对应输出空数组 []。

## 输出 JSON Schema
{
  "entities": [
    {"name": "简短名称", "label": "类型", "description": "可选，原文内容或特色说明"}
  ],
  "relations": [
    {"head": "头实体name", "tail": "尾实体name", "type": "关系类型"}
  ]
}

## Few-shot 示例
输入切片：
"去三亚旅游，第一天强烈推荐去蜈支洲岛潜水（门票加船票168元）。晚上可以去第一市场逛逛，吃一顿正宗的清补凉。住宿的话，预算充足可以选择海棠湾的亚特兰蒂斯酒店，离免税店很近。"
输出：
{
  "entities": [
    {"name": "三亚", "label": "Destination"},
    {"name": "蜈支洲岛", "label": "Attraction", "description": "门票加船票168元，适合潜水"},
    {"name": "第一市场", "label": "Restaurant", "description": "晚上可以逛"},
    {"name": "清补凉", "label": "Food", "description": "正宗特色小吃"},
    {"name": "海棠湾", "label": "Destination"},
    {"name": "亚特兰蒂斯酒店", "label": "Hotel", "description": "预算充足推荐，离免税店近"}
  ],
  "relations": [
    {"head": "蜈支洲岛", "tail": "三亚", "type": "LOCATED_IN"},
    {"head": "第一市场", "tail": "三亚", "type": "LOCATED_IN"},
    {"head": "第一市场", "tail": "清补凉", "type": "SERVES_FOOD"},
    {"head": "亚特兰蒂斯酒店", "tail": "海棠湾", "type": "LOCATED_IN"},
    {"head": "三亚", "tail": "亚特兰蒂斯酒店", "type": "RECOMMENDED_HOTEL"}
  ]
}
"""