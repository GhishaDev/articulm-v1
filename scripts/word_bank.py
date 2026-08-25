"""Illustrative word bank for fixtures and the tiny-overfit dataset.

Development tooling. Surface tones are written out explicitly (already
sandhi-applied where relevant) because ArticuLM consumes frontend output and
does not do G2P itself. These pronunciations are hand-written approximations,
not real frontend output.
"""

from __future__ import annotations

from fixture_corpus import Word, en, zh

# -- Chinese ---------------------------------------------------------------

ZH_WORDS: tuple[Word, ...] = (
    zh("我们", (("wo", 3), ("men", 5))),
    zh("新", (("xin", 1),)),
    zh("好", (("hao", 3),)),
    zh("高", (("gao", 1),)),
    zh("你好", (("ni", 2), ("hao", 3))),
    zh("今天", (("jin", 1), ("tian", 1))),
    zh("天气", (("tian", 1), ("qi", 4))),
    zh("很好", (("hen", 3), ("hao", 3))),
    zh("请", (("qing", 3),)),
    zh("稍等", (("shao", 1), ("deng", 3))),
    zh("一下", (("yi", 2), ("xia", 4))),
    zh("谢谢", (("xie", 4), ("xie", 5))),
    zh("欢迎", (("huan", 1), ("ying", 2))),
    zh("大家", (("da", 4), ("jia", 1))),
    zh("收看", (("shou", 1), ("kan", 4))),
    zh("新闻", (("xin", 1), ("wen", 2))),
    zh("报道", (("bao", 4), ("dao", 4))),
    zh("根据", (("gen", 1), ("ju", 4))),
    zh("最新", (("zui", 4), ("xin", 1))),
    zh("公布", (("gong", 1), ("bu", 4))),
    zh("数据", (("shu", 4), ("ju", 4))),
    zh("市场", (("shi", 4), ("chang", 3))),
    zh("保持", (("bao", 3), ("chi", 2))),
    zh("稳定", (("wen", 3), ("ding", 4))),
    zh("增长", (("zeng", 1), ("zhang", 3))),
    zh("人工智能", (("ren", 2), ("gong", 1), ("zhi", 4), ("neng", 2))),
    zh("新能源", (("xin", 1), ("neng", 2), ("yuan", 2))),
    zh("高端", (("gao", 1), ("duan", 1))),
    zh("制造", (("zhi", 4), ("zao", 4))),
    zh("领域", (("ling", 3), ("yu", 4))),
    zh("表现", (("biao", 3), ("xian", 4))),
    zh("突出", (("tu", 1), ("chu", 1))),
    zh("银行", (("yin", 2), ("hang", 2))),
    zh("正常", (("zheng", 4), ("chang", 2))),
    zh("营业", (("ying", 2), ("ye", 4))),
    zh("向前", (("xiang", 4), ("qian", 2))),
    zh("行走", (("xing", 2), ("zou", 3))),
    zh("百分之", (("bai", 3), ("fen", 1), ("zhi", 1))),
    zh("十二", (("shi", 2), ("er", 4))),
    zh("点五", (("dian", 3), ("wu", 3))),
    zh("二零二六", (("er", 4), ("ling", 2), ("er", 4), ("liu", 4))),
    zh("年", (("nian", 2),)),
    zh("全年", (("quan", 2), ("nian", 2))),
    zh("销售额", (("xiao", 1), ("shou", 4), ("e", 2))),
    zh("预计", (("yu", 4), ("ji", 4))),
    zh("将", (("jiang", 1),)),
    zh("模型", (("mo", 2), ("xing", 2))),
    zh("服务器", (("fu", 2), ("wu", 4), ("qi", 4))),
    zh("上", (("shang", 4),)),
    zh("运行", (("yun", 4), ("xing", 2))),
    zh("的", (("de", 5),)),
    zh("在", (("zai", 4),)),
    zh("和", (("he", 2),)),
    zh("其中", (("qi", 2), ("zhong", 1))),
    zh("整体", (("zheng", 3), ("ti", 3))),
    zh("国内", (("guo", 2), ("nei", 4))),
    zh("上半年", (("shang", 4), ("ban", 4), ("nian", 2))),
    zh("移动", (("yi", 2), ("dong", 4))),
    zh("二", (("er", 4),)),
    zh("女士", (("nv", 3), ("shi", 4))),
    zh("绿色", (("lv", 4), ("se", 4))),
    zh("旅行", (("lv", 3), ("xing", 2))),
    zh("父亲", (("fu", 4), ("qin", 1))),
    zh("方面", (("fang", 1), ("mian", 4))),
    zh("需求", (("xu", 1), ("qiu", 2))),
)

# -- English ---------------------------------------------------------------
# Phonemes restricted to the illustrative inventory in fixture_corpus.

EN_WORDS: tuple[Word, ...] = (
    en("we", (("w", 1), ("iy", 1))),
    en("can", (("k", 0), ("ah", 0), ("n", 0))),
    en("move", (("m", 1), ("uw", 1), ("v", 1))),
    en("AI", (("ey", 1), ("ay", 1))),
    en("GPU", (("jh", 0), ("iy", 1), ("p", 0), ("iy", 1), ("y", 0), ("uw", 1))),
    en("new", (("n", 1), ("y", 1), ("uw", 1))),
    en("key", (("k", 1), ("iy", 1))),
    en("view", (("v", 1), ("y", 1), ("uw", 1))),
    en("me", (("m", 1), ("iy", 1))),
    en("up", (("ah", 1), ("p", 1))),
)

ZH_BY_TEXT = {word.text: word for word in ZH_WORDS}
EN_BY_TEXT = {word.text: word for word in EN_WORDS}


def w(text: str) -> Word:
    """Look up a word by surface text (Chinese first, then English)."""
    if text in ZH_BY_TEXT:
        return ZH_BY_TEXT[text]
    if text in EN_BY_TEXT:
        return EN_BY_TEXT[text]
    raise KeyError(f"unknown word {text!r}")
