"""生成柯南主题评测任务集、米花町档案夹具，并打印模型部署提示。"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent
FIXTURE_DIR = DATA_DIR / "agent-fixtures"

# 档案柜 .md 份数（任务 3 评测依赖；增删文件时请同步改 build_tasks）
FIXTURE_MD_COUNT = 10


def write_fixtures():
    """档案柜：离线可查的红黑 / APTX / 主线人物（评测关键数字以 aptx 档案为准）。"""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    (FIXTURE_DIR / "README.md").write_text(
        "米花町档案室 · 任务五本地检索夹具。\n\n"
        "本目录是毛利侦探事务所整理的主线速查柜：APTX4869、黑衣组织、红方（FBI/公安/CIA）、"
        "宫野家、赤井家、毛利家与少年侦探团等。\n\n"
        "## 怎么用\n\n"
        "- 用 `file_search` 按文件名或关键词（如 APTX4869、琴酒、赤井、波本）检索\n"
        "- 年龄演算以 `aptx4869_dossier.md` 中的加粗数字为准（稳定评测）\n"
        "- 更细剧情、声优、集数请再查 `conan_wiki`（柯南中文 Wiki）\n\n"
        "## 目录索引\n\n"
        "| 文件 | 内容 |\n"
        "| --- | --- |\n"
        "| aptx4869_dossier.md | 药物机理、新一/灰原年龄差 |\n"
        "| black_org_roster.md | 黑衣组织主线干部与行动组 |\n"
        "| red_side_roster.md | FBI / 公安 / CIA 等对抗势力 |\n"
        "| detective_allies.md | 毛利家、博士、少年侦探团 |\n"
        "| miyano_line.md | 宫野家与雪莉线 |\n"
        "| akai_furuya_line.md | 赤井家、降谷零、世良真纯 |\n"
        "| org_boss_and_rum.md | 「那个人」与朗姆线笔记 |\n"
        "| main_plot_notes.md | 主线事件时间线摘要 |\n"
        "| cipher_todo.md | 待办暗号与保密提醒 |\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "aptx4869_dossier.md").write_text(
        "# APTX4869 档案（事务所速查）\n\n"
        "> 加粗年龄数字为本任务夹具约定，供计算器/沙箱题稳定作答；"
        "与动画具体集数有出入时，以本文件为准，wiki 作补充。\n\n"
        "## 基本资料\n\n"
        "- 全称缩写：APTX4869（Apoptoxin）\n"
        "- 编号谐音：4869 ≈ 日语「夏洛克 / Sherlock」读音梗\n"
        "- 组织内部别称线索：曾与「银色子弹 / Silver Bullet」研究线关联（宫野夫妇旧案）\n"
        "- 外观常见描述：印有 APTX4869 字样的胶嚢（白/红等说法见各版）\n"
        "- 研发者：宫野志保（组织代号 **Sherry / 雪莉**；缩小后化名 **灰原哀**）\n"
        "- 组织原意图：高效暗杀（诱导细胞坏死）；未成年体偶发「身体缩小」副作用\n\n"
        "## 确认缩小案例（评测用数字）\n\n"
        "| 人物 | 本名/化名 | 服用时真实年龄 | 外表年龄 | 年龄差 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 工藤新一 | 江户川柯南 | **17** | **7** | **10** |\n"
        "| 宫野志保 | 灰原哀 | **18** | **7** | **11** |\n\n"
        "- 年龄差（新一）：17 - 7 = **10** 岁\n"
        "- 年龄差（灰原）：18 - 7 = **11** 岁\n"
        "- 两人真实年龄差：18 - 17 = **1** 岁\n\n"
        "## 药效与解毒线索（剧情摘要）\n\n"
        "- 正常致死路径：细胞凋亡 / 组织希望「死无对证」\n"
        "- 意外路径：身体退行至幼童外观，智力与记忆大致保留\n"
        "- 临时解毒相关：白干儿（高纯度酒类）等可短暂缓解（剧中多次使用，有抗药性风险）\n"
        "- 灰原在阿笠研究所持续研制解药；资料曾遭组织销毁，研发极困难\n"
        "- 感冒药、某些药物可能干扰临时解毒效果（案件中多次成为关键变数）\n\n"
        "## 与红黑双方的关系\n\n"
        "- 黑衣组织用 APTX4869 处理「麻烦人物」；新一在游乐场附近被琴酒强迫服用\n"
        "- 灰原因姐姐宫野明美之死反抗组织，服毒自尽未成反而缩小，后投奔柯南一侧\n"
        "- 贝尔摩德等人似乎掌握比普通干部更多关于「未死之人」的信息\n"
        "- 检索关键词：APTX4869、Apoptoxin、银色子弹、缩小、解药、白干儿\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "black_org_roster.md").write_text(
        "# 黑衣组织 · 主线人物档案\n\n"
        "> 组织成员多用酒类代号。以下为与柯南主线强相关的公开情报摘要。\n\n"
        "## 行动与暗杀核心\n\n"
        "### 琴酒（Gin）\n\n"
        "- 职位：组织高层行动负责人之一，执行暗杀与灭口\n"
        "- 特征：长银发、黑衣、常驾车；性格冷酷多疑\n"
        "- 主线：强迫工藤新一服用 APTX4869 的关键人物；多次接近柯南却未完全识破\n"
        "- 搭档：伏特加\n\n"
        "### 伏特加（Vodka）\n\n"
        "- 职位：琴酒的行动搭档\n"
        "- 特征：墨镜、体型壮硕；相对听命于琴酒\n"
        "- 主线：多次与琴酒共同出现在组织行动现场\n\n"
        "### 基安蒂（Chianti） / 科恩（Korn）\n\n"
        "- 职位：狙击组\n"
        "- 主线：参与对目标的远程清除；与琴酒组有行动配合\n\n"
        "## 情报与特殊权限\n\n"
        "### 贝尔摩德（Vermouth）\n\n"
        "- 代号酒名：苦艾酒 / 贝尔摩德\n"
        "- 特征：擅长易容与演技；与「那个人」关系特殊；知晓大量秘密\n"
        "- 主线：知道新一/柯南、灰原等关键真相的危险人物；对工藤有司夫妇亦有纠葛\n"
        "- 备注：对组织内部部分人也保持信息不对等优势\n\n"
        "### 朗姆（Rum）\n\n"
        "- 职位：组织第二号人物相关线（「朗姆」）\n"
        "- 主线：近期主线重要谜题；外表/身份存在多重伪装嫌疑（档案见 org_boss_and_rum.md）\n\n"
        "## 潜入者与叛逃者（曾属组织）\n\n"
        "### 雪莉（Sherry）＝ 宫野志保 → 灰原哀\n\n"
        "- 职位：原组织科学家，APTX4869 研发者\n"
        "- 主线：姐姐宫野明美死后反抗，缩小逃出，现协助柯南；组织仍在追杀\n"
        "- 详见：miyano_line.md、aptx4869_dossier.md\n\n"
        "### 波本（Bourbon）＝ 安室透 / 降谷零\n\n"
        "- 表面：组织情报员「波本」\n"
        "- 真实：日本公安警察，潜入组织\n"
        "- 主线：与赤井秀一关系复杂（苏格兰事件等）；常以「安室透」身份接近毛利事务所\n"
        "- 详见：akai_furuya_line.md、red_side_roster.md\n\n"
        "### 苏格兰（Scotch）\n\n"
        "- 职位：已故组织成员；实为公安潜入员\n"
        "- 主线：与降谷零、赤井秀一三条线交织的悲剧节点\n\n"
        "### 基尔（Kir）＝ 水无沧美 / 本堂瑛海\n\n"
        "- 表面：组织成员；曾以记者「水无澪」等身份活动\n"
        "- 真实：CIA 特工（本堂瑛海）\n"
        "- 主线：赤井「死亡」疑云、组织高层任务的关键棋子；与红方有隐藏联系\n\n"
        "## 其他主线出现过的组织相关\n\n"
        "| 代号/名称 | 备注 |\n"
        "| --- | --- |\n"
        "| 宫野厚司 / 宫野艾蕾娜 | 灰原父母；科学家；与银色子弹研究相关；已故 |\n"
        "| 宫野明美 | 灰原姐姐；曾协助组织抢夺资金后欲脱身，被琴酒杀害 |\n"
        "| 匹斯可（Pisco）等 | 早期组织干部，案件中登场后出局 |\n"
        "| 乌丸莲耶 | 与组织最高层、「那个人」线索强相关的名字 |\n\n"
        "检索关键词：黑衣组织、琴酒、伏特加、贝尔摩德、雪莉、波本、朗姆、APTX4869。\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "red_side_roster.md").write_text(
        "# 红方 / 对抗黑衣组织势力 · 档案\n\n"
        "> 「红方」在粉丝语境中常指与组织对峙的 FBI、公安、CIA 及赤井家相关人员。"
        "柯南与灰原虽非特工，但是主线最前线的追查者。\n\n"
        "## FBI\n\n"
        "| 人物 | 角色摘要 |\n"
        "| --- | --- |\n"
        "| 赤井秀一 | FBI 王牌探员；曾潜入组织（代号诸星大）；神枪手；与贝尔摩德、琴酒多次交锋 |\n"
        "| 朱蒂·斯泰琳 (Jodie Starling) | FBI；赤井同事/情感线相关；曾以英语教师身份接近柯南一行 |\n"
        "| 詹姆斯·布莱克 (James Black) | FBI 搜查官；行动指挥角色之一 |\n"
        "| 安德烈·卡迈尔 (Andre Camel) | FBI；体型壮、行动组成员 |\n"
        "| 赤井务武 | 赤井秀一之父；与 MI6 等国际线相关（家庭见 akai_furuya_line.md） |\n\n"
        "## 日本公安\n\n"
        "| 人物 | 角色摘要 |\n"
        "| --- | --- |\n"
        "| 降谷零 | 公安；化名安室透；潜入组织代号波本；零之执行人相关传说 |\n"
        "| 风见裕也 | 公安；降谷零的部下/协助者 |\n"
        "| 苏格兰 | 已故公安潜入员（见黑衣组织档案） |\n\n"
        "## CIA\n\n"
        "| 人物 | 角色摘要 |\n"
        "| --- | --- |\n"
        "| 本堂瑛海 | CIA；组织内代号基尔（Kir）；化名水无澪等 |\n"
        "| 本堂瑛祐 | 与基尔线相关的家庭成员 |\n\n"
        "## 前线追查者（非特工编制）\n\n"
        "| 人物 | 角色摘要 |\n"
        "| --- | --- |\n"
        "| 工藤新一 / 江户川柯南 | 被 APTX4869 缩小后，以小学生身份追查组织 |\n"
        "| 灰原哀 | 叛逃科学家；提供组织与药物情报；协助研制解药 |\n"
        "| 世良真纯 | 女侦探；赤井家相关；行动力强 |\n"
        "| 玛丽世良 | 世良真纯之母；赤井家；外表曾受药物影响相关谜题 |\n"
        "| 冲矢昴 | 赤井秀一伪装身份之一；租住工藤家，暗中保护 |\n\n"
        "检索关键词：赤井、FBI、公安、降谷、安室、基尔、红方、冲矢昴。\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "detective_allies.md").write_text(
        "# 毛利家 · 博士 · 少年侦探团（友方日常线）\n\n"
        "## 毛利侦探事务所\n\n"
        "### 江户川柯南（工藤新一）\n\n"
        "- 真实身份：高中生侦探工藤新一，服用 APTX4869 后缩小\n"
        "- 寄住：毛利家；对外称「远房亲戚」\n"
        "- 手段：麻醉手表、变声蝶领带，借毛利小五郎之口破案\n"
        "- 目标：找回身体、揭露黑衣组织、保护身边人\n\n"
        "### 毛利兰\n\n"
        "- 身份：工藤新一青梅竹马；空手道高手\n"
        "- 主线：不知柯南即新一（长期）；多次陷入组织边缘危机\n"
        "- 备注：与新一的约定、话剧《灰色的传说》等情感线名场面\n\n"
        "### 毛利小五郎\n\n"
        "- 身份：「沉睡的小五郎」；前刑警，现私家侦探\n"
        "- 主线：常被柯南借刀破案；也因此被组织/贝尔摩德视线扫到\n"
        "- 家庭：与妃英理分居但未离婚；女儿毛利兰\n\n"
        "### 妃英理\n\n"
        "- 身份：律师；兰的母亲；小五郎之妻\n"
        "- 备注：与工藤有希子同期好友；推理能力不弱\n\n"
        "## 阿笠研究所\n\n"
        "### 阿笠博士\n\n"
        "- 身份：发明家；少数知道新一/灰原真实身份的大人\n"
        "- 作用：提供眼镜、手表、滑板等道具；收留灰原；研制相关科技支援\n\n"
        "### 灰原哀\n\n"
        "- 见 miyano_line.md 与 aptx4869_dossier.md\n"
        "- 日常：帝丹小学一年 B 班；少年侦探团成员；性格冷静毒舌\n\n"
        "## 少年侦探团\n\n"
        "| 成员 | 备注 |\n"
        "| --- | --- |\n"
        "| 江户川柯南 | 实质领导/破案核心 |\n"
        "| 灰原哀 | 智力担当；组织知识库 |\n"
        "| 吉田步美 | 活泼；对柯南有好感 |\n"
        "| 圆子（圆谷光彦） | 眼镜；喜欢讲道理 |\n"
        "| 小岛元太 | 力气大；喜欢吃 |\n\n"
        "## 帝丹高中相关\n\n"
        "| 人物 | 备注 |\n"
        "| --- | --- |\n"
        "| 铃木园子 | 兰的好友；铃木财团；常卷入案件 |\n"
        "| 世良真纯 | 转学生；侦探；赤井家 |\n"
        "| 白鸟任三郎 / 目暮十三 等 | 警视厅侧，常与小五郎/柯南联动（非红黑编制） |\n\n"
        "检索关键词：毛利、柯南、少年侦探团、阿笠、兰、小五郎。\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "miyano_line.md").write_text(
        "# 宫野家与雪莉线\n\n"
        "## 家族关系\n\n"
        "```text\n"
        "宫野厚司 —— 宫野艾蕾娜\n"
        "      |\n"
        " ┌────┴────┐\n"
        "宫野明美   宫野志保\n"
        "（已故） （Sherry → 灰原哀）\n"
        "```\n\n"
        "## 宫野厚司 / 宫野艾蕾娜\n\n"
        "- 身份：科学家；与黑衣组织有研究合作\n"
        "- 研究线索：银色子弹（Silver Bullet）—— 与 APTX4869 前身/相关项目有关\n"
        "- 结局：两人均已故；给志保留下录音等关键遗产情报\n\n"
        "## 宫野明美\n\n"
        "- 身份：志保的姐姐；曾为组织做事（含抢夺资金等任务）\n"
        "- 主线：希望带妹妹脱离组织；在交易后被琴酒杀害\n"
        "- 影响：直接促使宫野志保反抗组织并服下 APTX4869\n"
        "- 化名线索：曾使用「广田雅美」等名字活动\n\n"
        "## 宫野志保 / 雪莉 / 灰原哀\n\n"
        "- 组织代号：Sherry（雪莉）\n"
        "- 成就：完善/研发 APTX4869\n"
        "- 缩小后化名：灰原哀（名字取自推理小说角色名拼接梗）\n"
        "- 年龄夹具：真实 **18**，外表约 **7**，差 **11** 岁（见 aptx4869_dossier.md）\n"
        "- 现状：寄居阿笠家；协助柯南追查组织并研发解药；被组织列为必须清除对象\n"
        "- 性格：冷静、自嘲、外冷内热；对「寿命」与「连累他人」极度敏感\n\n"
        "## 与红黑交叉点\n\n"
        "- 黑方：琴酒灭口明美；组织追杀雪莉\n"
        "- 红方：灰原成为柯南最重要的内部情报来源之一\n"
        "- 贝尔摩德：对宫野家/雪莉线似乎另有关注\n\n"
        "检索关键词：宫野志保、灰原哀、雪莉、Sherry、宫野明美、银色子弹、APTX4869。\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "akai_furuya_line.md").write_text(
        "# 赤井家 · 降谷零 · 世良线\n\n"
        "## 赤井家谱（简化）\n\n"
        "```text\n"
        "赤井务武 —— 玛丽（Mary）\n"
        "      |\n"
        " ┌────┼─────────┐\n"
        "赤井秀一  赤井秀吉  世良真纯\n"
        "（FBI）  （麻将士） （女侦探）\n"
        "```\n\n"
        "## 赤井秀一\n\n"
        "- 阵营：**FBI**（不是黑衣组织）\n"
        "- 潜入经历：以「诸星大」身份加入组织，后身份暴露\n"
        "- 能力：枪法极准；推理与潜伏能力顶尖\n"
        "- 伪装：冲矢昴（寄住工藤家附近，监视/保护）等\n"
        "- 对手与纠葛：琴酒、贝尔摩德、降谷零（既冲突也合作）\n"
        "- 「死亡」疑云：曾上演被基尔枪击的戏码，实为骗局的一部分（主线关键转折）\n\n"
        "## 世良真纯\n\n"
        "- 身份：私立侦探；帝丹高中生；赤井秀一的妹妹\n"
        "- 特征：摩托车、行动派、自称要追赶「那个男人」（哥哥）的背影\n"
        "- 主线：多次与柯南并肩；逐步牵出赤井家与玛丽线\n\n"
        "## 玛丽世良\n\n"
        "- 身份：赤井秀一、秀吉、真纯之母\n"
        "- 备注：与 MI6 / 务武国际线相关；外表年龄异常是主线谜题之一（药物相关猜测常见于讨论，细节以 wiki 为准）\n\n"
        "## 赤井秀吉\n\n"
        "- 身份：职业麻将士；赤井家长男/次子设定中的一名兄弟\n"
        "- 备注：相对少直接打组织正面，但属赤井家情报网一环\n\n"
        "## 降谷零 / 安室透 / 波本\n\n"
        "- 公安真名：降谷零\n"
        "- 私家侦探/咖啡师伪装：安室透（常出入毛利事务所）\n"
        "- 组织代号：波本（Bourbon）\n"
        "- 立场：红方（公安），同时必须在组织内维持伪装\n"
        "- 与赤井：因苏格兰之死等原因长期敌视，后来情势复杂化\n"
        "- 能力：驾驶、格斗、推理、情报；「零の執行人」形象来源之一\n\n"
        "## 三角关系记忆钩\n\n"
        "- 赤井秀一（FBI） ↔ 降谷零（公安） ↔ 苏格兰（故）\n"
        "- 波本任务常包括：调查「冲矢昴」、接近柯南/灰原周边\n"
        "- 检索关键词：赤井秀一、冲矢昴、诸星大、降谷零、安室透、波本、世良真纯、玛丽。\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "org_boss_and_rum.md").write_text(
        "# 「那个人」与朗姆线 · 笔记\n\n"
        "> 此文件记录主线谜题方向，不做剧透式「最终定论」；"
        "具体身份争议请用 conan_wiki 核对最新章节。\n\n"
        "## 组织最高层\n\n"
        "- 成员口中的「那个人」：黑衣组织的最高领导者\n"
        "- 关联姓名线索：乌丸莲耶（常在讨论中与最高层挂钩）\n"
        "- 特征传闻：极高龄却保持影响力；与贝尔摩德关系非同一般\n\n"
        "## 朗姆（Rum）\n\n"
        "- 职位定位：组织序列中极高的二号人物线\n"
        "- 外貌传闻：有「假眼 / 义眼」「行动不便」等互相矛盾的目击描述\n"
        "- 调查意义：朗姆线推动近期主线；多名角色被怀疑与朗姆身份有关\n"
        "- 对柯南侧：朗姆的调查会波及公安、FBI 与毛利事务所周边\n\n"
        "## 贝尔摩德的特殊地位\n\n"
        "- 不属于普通「酒名干部」层级叙事：更像拥有直达最高层的管道\n"
        "- 掌握新一未死、灰原身份等极度敏感情报\n"
        "- 与工藤有司、工藤有希子（新一父母）有历史纠缠\n\n"
        "## 新一父母（辅助条目）\n\n"
        "| 人物 | 备注 |\n"
        "| --- | --- |\n"
        "| 工藤有司 | 推理小说作家；知晓儿子缩小；与组织/贝尔摩德周旋 |\n"
        "| 工藤有希子 | 前女星；易容高手；同样知晓并协助隐瞒 |\n\n"
        "检索关键词：那个人、乌丸、朗姆、Rum、贝尔摩德、工藤有司、工藤有希子。\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "main_plot_notes.md").write_text(
        "# 主线事件摘要（红黑 / APTX）\n\n"
        "> 按因果顺序的极简时间线，便于 Agent 检索「先发生了什么」。\n\n"
        "1. **游乐场事件**：工藤新一跟踪黑衣组织交易，被琴酒发现，强迫服下 APTX4869，身体缩小。\n"
        "2. **江户川柯南诞生**：新一化名寄住毛利家，开始双重生活。\n"
        "3. **宫野明美之死**：灰原的姐姐欲脱离组织，被琴酒杀害。\n"
        "4. **雪莉叛逃**：宫野志保服 APTX4869 未死反缩小，逃出后化名灰原哀，被阿笠博士收留。\n"
        "5. **解药研发**：灰原与博士尝试逆转缩小；多次临时恢复（白干儿等）推动关键案件。\n"
        "6. **FBI 登场**：赤井秀一、朱蒂等与组织正面冲突升级。\n"
        "7. **赤井「死亡」与复活布局**：涉及基尔（CIA）、组织信任测试与伪装战。\n"
        "8. **波本接近事务所**：降谷零以安室透身份贴身观察毛利/柯南/灰原。\n"
        "9. **冲矢昴入住**：赤井伪装邻近监视，红方内部张力上升。\n"
        "10. **世良 / 玛丽线**：赤井家更多成员进入主舞台。\n"
        "11. **朗姆线推进**：组织二号人物调查成为新主线引擎。\n\n"
        "## 给 Agent 的解题提示\n\n"
        "- 问年龄/药物/年龄差 → 先 `file_search` aptx4869_dossier，再 `calculator`\n"
        "- 问琴酒、贝尔摩德、雪莉 → black_org_roster 或 miyano_line\n"
        "- 问赤井是否组织成员 → red_side_roster：**FBI**，不是黑衣组织\n"
        "- 问安室透真实身份 → 公安·降谷零，组织代号波本\n"
        "- 需要剧集级细节 → `conan_wiki`\n\n"
        "检索关键词：主线、时间线、游乐场、叛逃、FBI、波本、朗姆、APTX4869。\n",
        encoding="utf-8",
    )

    (FIXTURE_DIR / "cipher_todo.md").write_text(
        "# 暗号草稿 TODO\n\n"
        "- TODO: 核对 APTX4869 与「夏洛克」谐音是否写进结案报告。\n"
        "- TODO: 红黑双方名单勿泄露给贝尔摩德线相关人员。\n"
        "- TODO: 朗姆目击描述互相矛盾，单独建疑点表（假眼 / 轮椅 / 老管家等说法）。\n"
        "- TODO: 赤井秀一「诸星大 / 冲矢昴」伪装时间线与波本调查重叠段要标注。\n"
        "- TODO: 灰原解药实验记录仅存阿笠研究所，档案柜只留摘要防泄密。\n",
        encoding="utf-8",
    )


def build_tasks():
    """28 题：单工具 + 红黑/APTX/档案多步；后半偏计算。答案尽量可由夹具或稳定算术推出。"""
    return [
        # ---- 1–10：原核心题（略调表述）----
        {
            "id": 1,
            "task": (
                "【真相计算】APTX 档案里新一真实年龄 17、外表 7。"
                "请计算 17 - 7，并告诉我 4869 乘以这个年龄差等于多少。"
            ),
            "expected_tools": ["calculator"],
            "expected_answer_contains": ["10", "48690"],
        },
        {
            "id": 2,
            "task": (
                "【博士沙箱】用 Python 打印：灰原真实年龄 18、外表 7 的年龄差；"
                "再打印新一 17 与灰原 18 的真实年龄差。两行数字即可。"
            ),
            "expected_tools": ["python_sandbox"],
            "expected_answer_contains": ["11", "1"],
        },
        {
            "id": 3,
            "task": "在 data/agent-fixtures 档案柜里找出所有 .md 文件，统计一共有几份。",
            "expected_tools": ["file_search"],
            "expected_answer_contains": [[str(FIXTURE_MD_COUNT), "十"]],
        },
        {
            "id": 4,
            "task": (
                "名侦探柯南取名致敬现实世界作者。"
                "请用维基百科查「柯南·道尔」或「阿瑟·柯南·道尔」，"
                "他最著名的侦探角色是谁？"
            ),
            "expected_tools": ["wiki"],
            "expected_answer_contains": [["福尔摩斯", "Holmes", "Sherlock"]],
        },
        {
            "id": 5,
            "task": (
                "先查柯南百科「灰原哀」（或「宫野志保」），确认她与 APTX4869 的关系；"
                "再根据档案约定：真实 18、外表 7，计算她缩小了多少岁。"
            ),
            "expected_tools": ["conan_wiki", "calculator"],
            "expected_answer_contains": [
                ["APTX", "雪莉", "Sherry", "宫野", "研发", "发明"],
                "11",
            ],
        },
        {
            "id": 6,
            "task": (
                "用 Python 写函数判断回文暗号，测试 'level' 和 'vodka'："
                "哪个是回文？输出两个词及其 True/False。"
            ),
            "expected_tools": ["python_sandbox"],
            "expected_answer_contains": ["level", "True", "vodka", "False"],
        },
        {
            "id": 7,
            "task": (
                "在 data/agent-fixtures 下查找所有内容包含 'APTX4869' 的档案路径。"
            ),
            "expected_tools": ["file_search"],
            "expected_answer_contains": ["aptx4869_dossier.md"],
        },
        {
            "id": 8,
            "task": (
                "计算 sqrt(4869) 的小数点后至少 4 位；"
                "可用计算器或 Python。"
            ),
            "expected_tools": ["calculator", "python_sandbox"],
            "expected_answer_contains": [["69.778", "69.78"]],
        },
        {
            "id": 9,
            "task": (
                "查柯南百科「赤井秀一」，他所属阵营是 FBI 还是黑衣组织？"
                "再用档案柜关键词「赤井」确认红方名单里是否有他。"
            ),
            "expected_tools": ["conan_wiki", "file_search"],
            "expected_answer_contains": [["FBI", "赤井"], ["红", "FBI"]],
        },
        {
            "id": 10,
            "task": (
                "读取 data/agent-fixtures/README.md，告诉我它第一段在说什么。"
            ),
            "expected_tools": ["file_search"],
            "expected_answer_contains": ["米花町", "档案"],
        },
        # ---- 11–20：扩展题（吃透新档案）----
        {
            "id": 11,
            "task": (
                "根据档案柜：新一年龄差 10 岁、灰原年龄差 11 岁。"
                "请计算两者年龄差之和，以及 10 * 11。"
            ),
            "expected_tools": ["calculator"],
            "expected_answer_contains": ["21", "110"],
        },
        {
            "id": 12,
            "task": (
                "在 data/agent-fixtures 中搜索「琴酒」，"
                "他强迫谁服用了 APTX4869？请说出受害者本名。"
            ),
            "expected_tools": ["file_search"],
            "expected_answer_contains": [["工藤新一", "新一"]],
        },
        {
            "id": 13,
            "task": (
                "查档案：安室透的公安真名是什么？他在黑衣组织里的代号又是什么？"
            ),
            "expected_tools": ["file_search"],
            "expected_answer_contains": [["降谷零", "降谷"], ["波本", "Bourbon"]],
        },
        {
            "id": 14,
            "task": (
                "宫野明美和宫野志保是什么关系？志保的组织代号是什么？"
                "请结合档案柜作答。"
            ),
            "expected_tools": ["file_search"],
            "expected_answer_contains": [["姐", "姊妹", "姐妹", "姐姐"], ["Sherry", "雪莉"]],
        },
        {
            "id": 15,
            "task": (
                "用 Python 打印 1 到 10 的平方和（即 1²+…+10²），只要最终数字。"
            ),
            "expected_tools": ["python_sandbox"],
            "expected_answer_contains": ["385"],
        },
        {
            "id": 16,
            "task": (
                "查柯南百科「贝尔摩德」或档案中的贝尔摩德条目："
                "她是否知晓新一/柯南相关秘密？用一句话回答，并提到贝尔摩德。"
            ),
            "expected_tools": ["conan_wiki", "file_search"],
            "expected_answer_contains": [["贝尔摩德", "Vermouth"], ["秘密", "知晓", "知道", "新一", "柯南"]],
        },
        {
            "id": 17,
            "task": (
                "在档案柜搜索「冲矢昴」，他其实是谁的伪装？"
            ),
            "expected_tools": ["file_search"],
            "expected_answer_contains": [["赤井秀一", "赤井"]],
        },
        {
            "id": 18,
            "task": (
                "主线摘要里：游乐场事件之后，新一化名寄住在哪一户人家？"
                "请从 data/agent-fixtures 作答。"
            ),
            "expected_tools": ["file_search"],
            "expected_answer_contains": [["毛利", "毛利家"]],
        },
        {
            "id": 19,
            "task": (
                "查柯南百科「琴酒」，确认他属于黑衣组织；"
                "再计算组织最爱的数字梗：4+8+6+9 等于多少？"
            ),
            "expected_tools": ["conan_wiki", "calculator"],
            "expected_answer_contains": [["黑", "组织", "Gin", "琴酒"], "27"],
        },
        {
            "id": 20,
            "task": (
                "档案提到 APTX4869 与「夏洛克」谐音有关。"
                "请先用 file_search 找到写有该谐音说明的档案，"
                "再用维基百科查「夏洛克·福尔摩斯」或「Sherlock Holmes」，"
                "说出他的搭档（助手）常见中文译名。"
            ),
            "expected_tools": ["file_search", "wiki"],
            "expected_answer_contains": [
                ["APTX4869", "夏洛克", "Sherlock", "4869"],
                ["华生", "Watson", "约翰"],
            ],
        },
        # ---- 21–28：加料计算题（APTX / 年龄 / 代号算术）----
        {
            "id": 21,
            "task": (
                "【计算】药物编号各位连乘：计算 4 * 8 * 6 * 9，并告诉我结果。"
            ),
            "expected_tools": ["calculator"],
            "expected_answer_contains": ["1728"],
        },
        {
            "id": 22,
            "task": (
                "【计算】新一差 10 岁、灰原差 11 岁。请计算 10² + 11²。"
            ),
            "expected_tools": ["calculator"],
            "expected_answer_contains": ["221"],
        },
        {
            "id": 23,
            "task": (
                "【计算】用计算器或 Python 求 17² - 7²"
                "（可理解为新一真实年龄平方减外表年龄平方）。"
            ),
            "expected_tools": ["calculator", "python_sandbox"],
            "expected_answer_contains": ["240"],
        },
        {
            "id": 24,
            "task": (
                "【计算】灰原真实 18、外表 7：求 18² - 7²。"
            ),
            "expected_tools": ["calculator"],
            "expected_answer_contains": ["275"],
        },
        {
            "id": 25,
            "task": (
                "先从档案确认新一 17/7、灰原 18/7，"
                "再计算四人数字之和：17+7+18+7。"
            ),
            "expected_tools": ["file_search", "calculator"],
            "expected_answer_contains": ["49"],
        },
        {
            "id": 26,
            "task": (
                "【计算】4869 除以 17 的商（整数除法，不要余数），"
                "以及 4869 对 17 取余。两个数都要答出。"
            ),
            "expected_tools": ["calculator", "python_sandbox"],
            "expected_answer_contains": ["286", "7"],
        },
        {
            "id": 27,
            "task": (
                "用 Python 计算：sum(range(1, 18)) 与 sum(range(1, 8))，"
                "并打印两者的差（即 1..17 的和减去 1..7 的和）。"
            ),
            "expected_tools": ["python_sandbox"],
            "expected_answer_contains": ["125"],
        },
        {
            "id": 28,
            "task": (
                "【计算】求 log10(4869) 的小数点后至少 3 位"
                "（可用 calculator 的 log10，或 Python math.log10）。"
            ),
            "expected_tools": ["calculator", "python_sandbox"],
            "expected_answer_contains": [["3.687", "3.69"]],
        },
    ]


def main():
    write_fixtures()
    md_files = sorted(FIXTURE_DIR.glob("*.md"))
    assert len(md_files) == FIXTURE_MD_COUNT, (
        f"夹具 .md 数量 {len(md_files)} != FIXTURE_MD_COUNT={FIXTURE_MD_COUNT}"
    )

    tasks = build_tasks()
    out = DATA_DIR / "tasks.json"
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"已生成评测任务集：{out.relative_to(DATA_DIR.parent)}（{len(tasks)} 题）")
    print(f"已生成柯南档案夹具：{FIXTURE_DIR.relative_to(DATA_DIR.parent)}（{len(md_files)} 份 .md）")
    for p in md_files:
        print(f"  - {p.name}")

    print("\n--- 模型部署提示（双环境）---")
    print("Agent 环境 llm_pj5:  pip install -r requirements.txt")
    print("vLLM  环境 vllm_pj5: bash scripts/setup_vllm_env.sh")
    print("  conda activate vllm_pj5 && bash scripts/serve_vllm.sh")
    print("  # 客户端默认 OPENAI_BASE_URL=http://localhost:8000/v1")
    print("  #          OPENAI_MODEL=Qwen/Qwen2.5-7B-Instruct")
    print("\n备选：Ollama")
    print("  ollama pull qwen2.5:7b-instruct && ollama serve")
    print("  # export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_MODEL=qwen2.5:7b-instruct")


if __name__ == "__main__":
    main()
