import os

from PIL import Image

edge = 'BadApple_only_edges'
frame = 'BadApple_pro_frames'

edge_add = 'BadApple_add'
if not os.path.exists(edge_add):
    os.makedirs(edge_add)

order = [int(i.strip(".jpg")) for i in os.listdir(edge) if i.endswith(".jpg")]
jpglist = [f"{i}.jpg" for i in sorted(order)]  # 直接读取可能非顺序帧

for i, jpg in enumerate(jpglist):
    img_e = Image.open(f'{edge}/{jpg}')
    img_p = Image.open(f'{frame}/{jpg}')
    w, h = img_p.size

    img_n = Image.new(mode='RGB', size=(w*2, h))
    img_n.paste(img_e, box=(0, 0))
    img_n.paste(img_p, box=(w, 0))
    img_n.save(f'{edge_add}/{i + 1}.jpg')

    print(f'{i + 1} / {len(jpglist)}')
