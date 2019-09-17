import os
from PIL import Image
from tqdm import tqdm

from pdb import set_trace


# STEP 1: Center Cropped to 562x562

src_dir = '/data/workplace/KDEF_and_AKDEF/KDEF'
dst_dir = '/data/workplace/KDEF_crop562'

if not os.path.exists(dst_dir):
	os.makedirs(dst_dir)

# TOTAL COUNTS: 2 series * 2 genders * 35 identities* 7 expressions * 3 (valid) angles = 2940
for folder in tqdm(sorted(os.listdir(src_dir))):
	for filename in sorted(os.listdir(src_dir+'/'+folder)):
		angle  = filename[6:-4]
		if angle not in ['HL', 'HR', 'S']:
			continue
		expression = filename[4:6]
		
		image = Image.open(src_dir+'/'+folder+'/'+filename)
		width, height = image.size
		half = (height - width) / 2
		image = image.crop((0, half, width, height-half))
		
		image.save(dst_dir+'/'+filename)
# RENAME: AF31V.JPG -> AF31SAHL.JPG
# ------: AM31H.JPG -> AM31SUHR.JPG

set_trace()

# STEP 2: Generate Pairs

src_dir = dst_dir
dst_dir = '/data/workplace/KDEF_pair3'

if not os.path.exists(dst_dir):
	os.makedirs(dst_dir)

for series in ['A', 'B']:
	for gender in ['F', 'M']:
		for identity in xrange(1, 36):
			base = series + gender + str(identity).zfill(2)
			for angle in ['HL', 'HR', 'S']:
				neutral = base + 'NE' + angle + '.JPG'
				im_ne  = Image.open(src_dir+'/'+neutral)
				for expression in ['AF', 'AN', 'DI', 'HA', 'SA', 'SU']:
					side = base + expression + angle + '.JPG'
					im_sd  = Image.open(src_dir+'/'+side)
					
					im_pair = Image.new('RGB', (562*2, 562))
					im_pair.paste(im_ne, (0, 0))
					im_pair.paste(im_sd, (562, 0))
					
					filename = base + 'NE' + expression + angle + '.JPG'
					im_pair.save(dst_dir+'/'+filename)

set_trace()

# STEP 3: Generate Embeddings

# See visualize.py