import cv2
root = '/home/hzp/codes/RAFT/aligncase_1_rgb/result/'
K1 = root + 'aligned_K1.png'
source = root + 'source_resize.png'
target = root + 'target_resize.png'
align = root + 'source_resize_refine_bycv2remap.png'

K1_img = cv2.imread(K1)
source_img = cv2.imread(source)
target_img = cv2.imread(target)
align_img = cv2.imread(align)

dif_s = cv2.absdiff(source_img, target_img) * 5
cv2.imwrite(root+'img_diff_source.png', cv2.cvtColor(dif_s,cv2.COLOR_BGR2GRAY))

dif_k = cv2.absdiff(K1_img, target_img) * 5
cv2.imwrite(root+'img_diff_K1.png', cv2.cvtColor(dif_k,cv2.COLOR_BGR2GRAY))

dif_n = cv2.absdiff(align_img, target_img) * 5
cv2.imwrite(root+'img_diff_raft.png', cv2.cvtColor(dif_n,cv2.COLOR_BGR2GRAY))
