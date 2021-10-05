import numpy as np
import argparse
import matplotlib.pyplot as plt
import string
import re
import math
import os
import torch
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
sns.set_theme(style="darkgrid")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.autolayout"] = True

parser = argparse.ArgumentParser(description = 'MSCRED evaluation')
parser.add_argument('--thred_broken', type = int, default = 0.005,
				   help = 'broken pixel thred')
parser.add_argument('--alpha', type = int, default = 1.5,
				   help = 'scale coefficient of max valid anomaly')
parser.add_argument('--valid_start_point',  type = int, default = 8000,
						help = 'test start point')
parser.add_argument('--valid_end_point',  type = int, default = 10000,
						help = 'test end point')
parser.add_argument('--test_start_point',  type = int, default = 10000,
						help = 'test start point')
parser.add_argument('--test_end_point',  type = int, default = 20000,
						help = 'test end point')
parser.add_argument('--gap_time', type = int, default = 10,
				   help = 'gap time between each segment')
parser.add_argument('--matrix_data_path', type = str, default = './data/matrix_data/',
				   help='matrix data path')

args = parser.parse_args()
print(args)

thred_b = args.thred_broken
alpha = args.alpha
gap_time = args.gap_time
valid_start = args.valid_start_point//gap_time
valid_end = args.valid_end_point//gap_time
test_start = args.test_start_point//gap_time
test_end = args.test_end_point//gap_time

valid_anomaly_score = np.zeros((valid_end - valid_start , 1))
test_anomaly_score = np.zeros((test_end - test_start, 1))

valid_anomaly_score_FGSM_linf = np.zeros((valid_end - valid_start , 1))
test_anomaly_score_FGSM_linf = np.zeros((test_end - test_start, 1))

valid_anomaly_score_FGSM_l1 = np.zeros((valid_end - valid_start , 1))
test_anomaly_score_FGSM_l1 = np.zeros((test_end - test_start, 1))

valid_anomaly_score_FGSM_l2 = np.zeros((valid_end - valid_start , 1))
test_anomaly_score_FGSM_l2 = np.zeros((test_end - test_start, 1))

valid_anomaly_score_PGD_linf = np.zeros((valid_end - valid_start , 1))
test_anomaly_score_PGD_linf = np.zeros((test_end - test_start, 1))

valid_anomaly_score_PGD_l2 = np.zeros((valid_end - valid_start , 1))
test_anomaly_score_PGD_l2 = np.zeros((test_end - test_start, 1))

valid_anomaly_score_SL1D = np.zeros((valid_end - valid_start , 1))
test_anomaly_score_SL1D = np.zeros((test_end - test_start, 1))

matrix_data_path = args.matrix_data_path
test_data_path = matrix_data_path + "test_data/"
reconstructed_data_path = matrix_data_path + "reconstructed_data/"
reconstructed_data_path_FGSM_linf = matrix_data_path + "reconstructed_data_FGSM_linf/"
reconstructed_data_path_FGSM_l1 = matrix_data_path + "reconstructed_data_FGSM_l1/"
reconstructed_data_path_FGSM_l2 = matrix_data_path + "reconstructed_data_FGSM_l2/"
reconstructed_data_path_PGD_linf = matrix_data_path + "reconstructed_data_PGD_linf/"
reconstructed_data_path_PGD_l2 = matrix_data_path + "reconstructed_data_PGD_l2/"
reconstructed_data_path_SL1D = matrix_data_path + "reconstructed_data_SL1D/"
#reconstructed_data_path = matrix_data_path + "matrix_pred_data/"

criterion = torch.nn.MSELoss()

for i in range(valid_start, test_end):
	path_temp_1 = os.path.join(test_data_path, "test_data_" + str(i) + '.npy')
	gt_matrix_temp = np.load(path_temp_1)

	path_temp_2 = os.path.join(reconstructed_data_path, "reconstructed_data_" + str(i) + '.npy')
	path_temp_2_FGSM_linf = os.path.join(reconstructed_data_path_FGSM_linf, "reconstructed_data_" + str(i) + '.npy')
	path_temp_2_FGSM_l1 = os.path.join(reconstructed_data_path_FGSM_l1, "reconstructed_data_" + str(i) + '.npy')
	path_temp_2_FGSM_l2 = os.path.join(reconstructed_data_path_FGSM_l2, "reconstructed_data_" + str(i) + '.npy')
	path_temp_2_PGD_linf = os.path.join(reconstructed_data_path_PGD_linf, "reconstructed_data_" + str(i) + '.npy')
	path_temp_2_PGD_l2 = os.path.join(reconstructed_data_path_PGD_l2, "reconstructed_data_" + str(i) + '.npy')
	path_temp_2_SL1D = os.path.join(reconstructed_data_path_SL1D, "reconstructed_data_" + str(i) + '.npy')


	#path_temp_2 = os.path.join(reconstructed_data_path, "pcc_matrix_full_test_" + str(i) + '_pred_output.npy')
	reconstructed_matrix_temp = np.load(path_temp_2)
	reconstructed_matrix_temp_FGSM_linf = np.load(path_temp_2_FGSM_linf)
	reconstructed_matrix_temp_FGSM_l1 = np.load(path_temp_2_FGSM_l1)
	reconstructed_matrix_temp_FGSM_l2 = np.load(path_temp_2_FGSM_l2)
	reconstructed_matrix_temp_PGD_linf = np.load(path_temp_2_PGD_linf)
	reconstructed_matrix_temp_PGD_l2 = np.load(path_temp_2_PGD_l2)
	reconstructed_matrix_temp_SL1D = np.load(path_temp_2_SL1D)



	# reconstructed_matrix_temp = np.transpose(reconstructed_matrix_temp, [0, 3, 1, 2])
	#print(reconstructed_matrix_temp.shape)
	#first (short) duration scale for evaluation  
	select_gt_matrix = np.array(gt_matrix_temp)[-1][0] #get last step matrix

	select_reconstructed_matrix = np.array(reconstructed_matrix_temp)[0][0]
	select_reconstructed_matrix_FGSM_linf = np.array(reconstructed_matrix_temp_FGSM_linf)[0][0]
	select_reconstructed_matrix_FGSM_l1 = np.array(reconstructed_matrix_temp_FGSM_l1)[0][0]
	select_reconstructed_matrix_FGSM_l2 = np.array(reconstructed_matrix_temp_FGSM_l2)[0][0]
	select_reconstructed_matrix_PGD_linf = np.array(reconstructed_matrix_temp_PGD_linf)[0][0]
	select_reconstructed_matrix_PGD_l2 = np.array(reconstructed_matrix_temp_PGD_l2)[0][0]
	select_reconstructed_matrix_SL1D = np.array(reconstructed_matrix_temp_SL1D)[0][0]

	#compute number of broken element in residual matrix
	select_matrix_error = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix))
	select_matrix_error_FGSM_linf = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix_FGSM_linf))
	select_matrix_error_FGSM_l1 = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix_FGSM_l1))
	select_matrix_error_FGSM_l2 = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix_FGSM_l2))
	select_matrix_error_PGD_linf = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix_PGD_linf))
	select_matrix_error_PGD_l2 = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix_PGD_l2))
	select_matrix_error_SL1D = np.square(np.subtract(select_gt_matrix, select_reconstructed_matrix_SL1D))

	num_broken = len(select_matrix_error[select_matrix_error > thred_b])
	num_broken_FGSM_linf = len(select_matrix_error_FGSM_linf[select_matrix_error_FGSM_linf > thred_b])
	num_broken_FGSM_l1 = len(select_matrix_error_FGSM_l1[select_matrix_error_FGSM_l1 > thred_b])
	num_broken_FGSM_l2 = len(select_matrix_error_FGSM_l2[select_matrix_error_FGSM_l2 > thred_b])
	num_broken_PGD_linf = len(select_matrix_error_PGD_linf[select_matrix_error_PGD_linf > thred_b])
	num_broken_PGD_l2 = len(select_matrix_error_PGD_l2[select_matrix_error_PGD_l2 > thred_b])
	num_broken_SL1D = len(select_matrix_error_SL1D[select_matrix_error_SL1D > thred_b])

	#print num_broken
	if i < valid_end:
		valid_anomaly_score[i - valid_start] = num_broken
	else:
		test_anomaly_score[i - test_start] = num_broken

	# print num_broken_FGSM_linf
	if i < valid_end:
		valid_anomaly_score_FGSM_linf[i - valid_start] = num_broken_FGSM_linf
	else:
		test_anomaly_score_FGSM_linf[i - test_start] = num_broken_FGSM_linf

	# print num_broken_FGSM__l1
	if i < valid_end:
		valid_anomaly_score_FGSM_l1[i - valid_start] = num_broken_FGSM_l1
	else:
		test_anomaly_score_FGSM_l1[i - test_start] = num_broken_FGSM_l1

	# print num_broken_FGSM_l2
	if i < valid_end:
		valid_anomaly_score_FGSM_l2[i - valid_start] = num_broken_FGSM_l2
	else:
		test_anomaly_score_FGSM_l2[i - test_start] = num_broken_FGSM_l2

	# print num_broken_PGD_linf
	if i < valid_end:
		valid_anomaly_score_PGD_linf[i - valid_start] = num_broken_PGD_linf
	else:
		test_anomaly_score_PGD_linf[i - test_start] = num_broken_PGD_linf

	# print num_broken_PGD_l2
	if i < valid_end:
		valid_anomaly_score_PGD_l2[i - valid_start] = num_broken_PGD_l2
	else:
		test_anomaly_score_PGD_l2[i - test_start] = num_broken_PGD_l2

	# print num_broken_SL1D
	if i < valid_end:
		valid_anomaly_score_SL1D[i - valid_start] = num_broken_SL1D
	else:
		test_anomaly_score_SL1D[i - test_start] = num_broken_SL1D

valid_anomaly_max = np.max(valid_anomaly_score.ravel())
valid_anomaly_max_FGSM_linf = np.max(valid_anomaly_score_FGSM_linf.ravel())
valid_anomaly_max_FGSM_l1 = np.max(valid_anomaly_score_FGSM_l1.ravel())
valid_anomaly_max_FGSM_l2 = np.max(valid_anomaly_score_FGSM_l2.ravel())
valid_anomaly_max_PGD_linf = np.max(valid_anomaly_score_PGD_linf.ravel())
valid_anomaly_max_PGD_l2 = np.max(valid_anomaly_score_PGD_l2.ravel())
valid_anomaly_max_SL1D = np.max(valid_anomaly_score_SL1D.ravel())

test_anomaly_score = test_anomaly_score.ravel()
test_anomaly_score_FGSM_linf = test_anomaly_score_FGSM_linf.ravel()
test_anomaly_score_FGSM_l1 = test_anomaly_score_FGSM_l1.ravel()
test_anomaly_score_FGSM_l2 = test_anomaly_score_FGSM_l2.ravel()
test_anomaly_score_PGD_linf = test_anomaly_score_PGD_linf.ravel()
test_anomaly_score_PGD_l2 = test_anomaly_score_PGD_l2.ravel()
test_anomaly_score_PGD_SL1D = test_anomaly_score_SL1D.ravel()

#print(test_anomaly_score)
# plot anomaly score curve and identification result
anomaly_pos = np.zeros(5)
root_cause_gt = np.zeros((5, 3))
anomaly_span = [30,90,60]
root_cause_f = open("./data/test_anomaly.csv", "r")
row_index = 0
for line in root_cause_f:
	line=line.strip()
	anomaly_axis = int(re.split(',',line)[0])
	anomaly_pos[row_index] = anomaly_axis/gap_time - test_start - anomaly_span[row_index%3]/gap_time
	#print(anomaly_pos[row_index])
	root_list = re.split(',',line)[1:]
	for k in range(len(root_list)-1):
		root_cause_gt[row_index][k] = int(root_list[k])
	row_index += 1
root_cause_f.close()

fig, axes = plt.subplots()
# fig = plt.figure()
# axes = fig.gca(projection='3d')
#plt.plot(test_anomaly_score, 'black', linewidth = 2)
test_num = test_end - test_start
# plt.xticks(fontsize = 25)
# plt.ylim((0, 100))
# plt.yticks(np.arange(0, 101, 20), fontsize = 25)


# plt.plot(test_anomaly_score_FGSM_l1, color = 'green',linestyle='dashed', linewidth = 2)
# plt.plot(test_anomaly_score_FGSM_l2, color = 'green',linestyle='dotted', linewidth = 2)

# plt.plot(test_anomaly_score_PGD_l2, color = 'blue',linestyle='dashed', linewidth = 2)



threshold = np.full((test_num), valid_anomaly_max * alpha)
threshold_FGSM_linf = np.full((test_num), valid_anomaly_max_FGSM_linf * alpha)
threshold_FGSM_l1 = np.full((test_num), valid_anomaly_max_FGSM_l1 * alpha)
threshold_FGSM_l2 = np.full((test_num), valid_anomaly_max_FGSM_l2 * alpha)
threshold_PGD_linf = np.full((test_num), valid_anomaly_max_PGD_linf * alpha)
threshold_PGD_l2 = np.full((test_num), valid_anomaly_max_PGD_l2 * alpha)
threshold_SL1D = np.full((test_num), valid_anomaly_max_SL1D * alpha)

plt.plot(test_anomaly_score, color = '#333333', linewidth = 2)
axes.plot(threshold, color = '#ff0000', linestyle = '--',linewidth = 2)

# plt.plot(test_anomaly_score_FGSM_linf-650, color = '#fcc438', linewidth = 1)
plt.plot(test_anomaly_score_FGSM_l1, color = '#fcc438', linewidth = 1)
# plt.plot(test_anomaly_score_FGSM_l2, color = '#fcc438', linewidth = 1)

# axes.plot(threshold_FGSM_linf-1120, color = '#834187', linestyle = '-.',linewidth = 2)

# plt.plot(test_anomaly_score_PGD_linf-650, color = '#0c7cba', linewidth = 1)
# plt.plot(test_anomaly_score_PGD_l1-650, color = '#0c7cba', linewidth = 1)
# plt.plot(test_anomaly_score_PGD_l2, color = '#0c7cba',linestyle = '--', linewidth = 1)
# axes.plot(threshold_PGD_linf-1120, color = '#0c7cba', linestyle = ':',linewidth = 2)

plt.plot(test_anomaly_score_SL1D, color = '#7ab648', linewidth = 1)
# axes.plot(threshold_SL1D, color = '#ef8d22', linestyle = '--',linewidth = 1)



from sklearn.metrics import classification_report

def get_classification_report_(anomaly_score,anomaly_max):
	result=[]
	for i in anomaly_score:
		if i>=anomaly_max * alpha:
			result.append(1)
		else:
			result.append(0)
	anomaly_list=np.zeros(len(anomaly_score))

	for k in range(len(anomaly_pos)):
		print(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/gap_time)
		anomaly_list[int(anomaly_pos[k]): int(anomaly_pos[k] + anomaly_span[k%3]/gap_time)]=1
		# axes.axvspan(anomaly_pos[k], anomaly_pos[k] + anomaly_span[k%3]/gap_time, color='#c92d39', linewidth=2)
	print(classification_report(anomaly_list.tolist(),result,digits=4))
  	# return
get_classification_report_(test_anomaly_score,valid_anomaly_max)
get_classification_report_(test_anomaly_score_FGSM_linf,valid_anomaly_max_FGSM_linf)
get_classification_report_(test_anomaly_score_PGD_linf,valid_anomaly_max_PGD_linf)
get_classification_report_(test_anomaly_score_SL1D,valid_anomaly_max_SL1D)
# axes.plot(xs=range(len(test_anomaly_score_FGSM_linf)), ys=test_anomaly_score_FGSM_linf, zs=0, zdir='y',color = '#834187', linewidth = 1)
# axes.plot(xs=range(len(threshold_FGSM_linf)), ys=threshold_FGSM_linf-450,zs=0, zdir='y', color = '#834187', linestyle = '--',linewidth = 1)
#
# plt.plot(test_anomaly_score_PGD_linf, color = '#0c7cba', linewidth = 1)
# axes.plot(threshold_PGD_linf, color = '#0c7cba', linestyle = '--',linewidth = 1)
#
# plt.plot(test_anomaly_score_SL1D, color = '#ef8d22', linewidth = 1)
# axes.plot(threshold_SL1D, color = '#ef8d22', linestyle = '--',linewidth = 1)
#
# plt.plot(test_anomaly_score, color = '#333333', linewidth = 2)
# axes.plot(threshold, color = '#333333', linestyle = '--',linewidth = 1)

for k in range(len(anomaly_pos)):
	if k==2:
		axes.axvspan(anomaly_pos[k]-2, anomaly_pos[k]-2 + anomaly_span[k%3]/gap_time, color='#d56872', linewidth=2)
	else:
		axes.axvspan(anomaly_pos[k] + 3, anomaly_pos[k] +3 + anomaly_span[k % 3] / gap_time, color='#d56872', linewidth=2)

# labels = [' ', '0e3', '2e3', '4e3', '6e3', '8e3', '10e3']
# axes.set_xticklabels(labels, rotation = 25, fontsize = 20)
plt.xlabel('Test Time', fontsize = 14, fontweight='bold')
plt.ylabel('Anomaly Score', fontsize = 14, fontweight='bold')
axes.spines['right'].set_visible(False)
axes.spines['top'].set_visible(False)
axes.yaxis.set_ticks_position('left')
axes.xaxis.set_ticks_position('bottom')
fig.subplots_adjust(bottom=0.25)
fig.subplots_adjust(left=0.25)
plt.title("MSCRED", size = 14, fontweight='bold')
# plt.legend(["Original","FGSM","PGD","Orginal_Threshold","FGSM_Threshold","PGD_Threshold"])
plt.legend([
			"No Attack","Threshold",
			"FGSM L1",
			# "FGSM L∞",#"FGSM L∞ Threshold",
			# "FGSM_l1",
			# "FGSM_l2",
			# "PGD L2",
			# "PGD L∞",
	#"PGD L∞ Threshold"
			# "PGD_l2",
			"SL1D",
	# "SL1D Threshold",


			# "FGSM_l1_Threshold",
			# "FGSM_l2_Threshold",

			# "PGD_l2_Threshold",
			"Anomaly"
			],loc='top center',ncol=2,fancybox=True)
fig.tight_layout()
# axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)
# plt.savefig('./outputs/anomaly_score_combine.jpg')
plt.show()




from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
# import matplotlib.pyplot as plt


# fig = plt.figure()
# ax1 = fig.gca(projection='3d')
# ax1.plot(xs=range(len(test_anomaly_score_FGSM_linf)),ys=test_anomaly_score_FGSM_linf, zdir='y',zs=0, color = '#834187', linewidth = 1)
# ax1.plot(xs=range(len(threshold_FGSM_linf)),ys=threshold_FGSM_linf,zs=0, color = '#834187',zdir='y', linestyle = '--',linewidth = 1)
#
# ax1.plot(xs=range(len(test_anomaly_score_SL1D)),ys=test_anomaly_score_SL1D,zs=0,zdir='y', color = '#ef8d22', linewidth = 1)
# ax1.plot(xs=range(len(threshold_SL1D)),ys=threshold_SL1D,zs=0, color = '#ef8d22',zdir='y', linestyle = '--',linewidth = 1)
#
# ax1.plot(xs=range(len(test_anomaly_score)),ys=test_anomaly_score,zs=0,zdir='y', color = '#333333', linewidth = 1)
# ax1.plot(xs=range(len(threshold)),ys=threshold,zs=0, color = '#333333',zdir='y', linestyle = '--',linewidth = 1)
#
# plt.show()


# fig = plt.figure()
# ax1 = fig.add_subplot(111, projection='3d')
# ax1.bar(left=range(len(test_anomaly_score_FGSM_linf)),height=test_anomaly_score_FGSM_linf,zs=0, color = '#834187')
# # ax1.plot(xs=range(len(threshold_FGSM_linf)),ys=threshold_FGSM_linf,zs=0, color = '#834187', linestyle = '--',linewidth = 1)
#
#
# ax1.bar(left=range(len(test_anomaly_score_SL1D)),height=test_anomaly_score_SL1D,zs=1, color = '#ef8d22')
# ax1.plot(xs=range(len(threshold_SL1D)),ys=threshold_SL1D,zs=1, color = '#ef8d22', linestyle = '--',linewidth = 1)
# for c, z in zip(['r', 'g', 'b', 'y'], [30, 20, 10, 0]):
#     xs = np.arange(20)
#     ys = np.random.rand(20)
#
#     # You can provide either a single color or an array. To demonstrate this,
#     # the first bar of each set will be colored cyan.
#     cs = [c] * len(xs)
#     cs[0] = 'c'
#     ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# plt.show()

#
# host = host_subplot(111, axes_class=AA.Axes)
# # plt.subplots_adjust(right=0.75)
#
# par1 = host.twinx()
# par2 = host.twinx()
# par3 = host.twinx()
#
# offset = 0
# new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#
# par1.axis["right"] = new_fixed_axis(loc="left",
#                                     axes=par1,
#                                     offset=(offset-150, 0))
# par2.axis["right"] = new_fixed_axis(loc="left",
#                                     axes=par2,
#                                     offset=(offset-100, 0))
# par3.axis["right"] = new_fixed_axis(loc="left",
#                                     axes=par2,
#                                     offset=(offset-50, 0))
#
# # par2.axis["right"].toggle(all=True)
#
# # host.set_xlim(0, 2)
# # host.set_ylim(0, 1350)
# par1.set_ylim(0, 1200)
# par2.set_ylim(0, 100)
# par3.set_ylim(0, 1350)
#
# # host.set_xlabel("Distance")
# # host.set_ylabel("")
# par1.set_ylabel("Temperature")
# par2.set_ylabel("Velocity")
# par3.set_ylabel("Density")
#
# p1, = par1.plot(test_anomaly_score_FGSM_linf, color = '#834187', linewidth = 1)
# par1.plot(threshold_FGSM_linf, color = '#834187', linestyle = '--',linewidth = 1)
#
# # axes.plot(threshold_PGD_linf, color = '#0c7cba', linestyle = '--',linewidth = 1)
# p2, = par2.plot(test_anomaly_score_SL1D, color = '#ef8d22', linewidth = 1)
# par2.plot(threshold_SL1D, color = '#ef8d22', linestyle = '--',linewidth = 1)
#
# p3, = par3.plot(test_anomaly_score, color = '#333333', linewidth = 2)
# par3.plot(threshold, color = '#333333', linestyle = '--',linewidth = 1)
#
#
#
# # par1.set_ylim(0, 4)
# # par2.set_ylim(1, 65)
# host.axis('off')
# host.legend()
#
# par1.axis["left"].label.set_color(p1.get_color())
# par2.axis["left"].label.set_color(p2.get_color())
# par3.axis["left"].label.set_color(p3.get_color())
#
# plt.draw()
# plt.show()


