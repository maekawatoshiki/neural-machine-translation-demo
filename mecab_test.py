import sys
import MeCab

m = MeCab.Tagger("-Owakati")

parsed = m.parse("明日は晴れそうですね").split()
print(parsed)

