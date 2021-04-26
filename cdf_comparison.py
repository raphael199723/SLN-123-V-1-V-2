import numpy as np
import pandas as pd
import sys, os, math
import matplotlib.pyplot as plt
file_name='DNN_and_FN_output_predict1.csv'
#file_name1='boosting_10time_mean_output_predict.csv'
#file_name2='boosting_output_predict.csv'
#file_name3='fully_10time_mean_output_predict.csv'
#file_name4='SLN_and_FN_output_predict.csv'
#file_name5='xboost_output_predict.csv'
#file_name6='xboost_wtsd_output_predict.csv'
#file_name7='XGBoost_and_FN_output_predict.csv'
file_name8='DNN_and_FN_output_predict.csv'
#file_name9='CNN_output_predict4.csv'
#file_name10='CNN_and_FN_output_predict10.csv'
#file_name11='CNN1D_and_FN_output_predict.csv'
#file_name12='CNN1d_output_predict.csv'


range_num = 350

def load_data(path=str):
	csv = pd.read_csv(path)
	return csv

def analysis_by_2_norm(csv=pd.core.frame.DataFrame):
	predict_value = []
	ground_truth = []
	error_value = [] # 2-norm
	for i in range(len(csv)):
		predict_value.append(csv['ans'][i].split(","))
		ground_truth.append(csv['label'][i].split(","))
		norm_2_square = ( float(predict_value[i][0])-float(ground_truth[i][0]) )**2 + ( float(predict_value[i][1])-float(ground_truth[i][1]) )**2
		error_value.append( math.sqrt(norm_2_square) )
	sum_M = sum(error_value)
	M = len(error_value)
	print('M:',M,';sum_M:',sum_M,';MDE:',sum_M/M,'cm')
	return error_value

def plt_cdf(path=str,data=list,name=str,range_num=int):
	plt.clf()
	predict_max_value = max(data)+2
	print('max error:',max(data))
	if predict_max_value > range_num:
		range_num = int(predict_max_value)
	error_counter = []
	cnt = 0 
	for i in range(int(range_num)+1):
		for j in data:
			if j <= i:
				cnt+=1
		error_counter.append(cnt)
		cnt = 0
	print('max range:',i)
	error_max = max(error_counter)
	#print(error_counter)
	print(error_max)

	#print(error)
	print(path+':')
	print('Now, we analysis',name,', the total number of data is',len(data),'.')
	print('The number of errors within 1 centimeter is',error_counter[1],'.')
	print('The number of errors within 2 centimeters is',error_counter[2],'.')
	print('The number of errors within 5 centimeters is',error_counter[5],'.')
	print('The number of errors within 10 centimeters is',error_counter[10],'.')
	print('The number of errors within 20 centimeters is',error_counter[20],'.')
	print('The number of errors within 50 centimeters is',error_counter[50],'.')
	print('The number of errors within 80 centimeters is',error_counter[80],'.')
	print('The number of errors within 100 centimeters is',error_counter[100],'.')
	print('The number of errors within 200 centimeters is',error_counter[200],'.')
	print('The number of errors within 300 centimeters is',error_counter[300],'.')
#	print('The number of errors within 400 centimeters is',error_counter[400],'.')
#	print('The number of errors within 500 centimeters is',error_counter[500],'.')
#	print('The number of errors within 600 centimeters is',error_counter[600],'.')
#	print('The number of errors within 700 centimeters is',error_counter[700],'.')
#	print('The number of errors within 800 centimeters is',error_counter[800],'.')
	print('The maximum error is',predict_max_value-2,'centimeters .')
	print('the index of half value:',int((len(data)+1)/2)-1)
	print('The error value of CDF 0.5 is',sorted(data)[int((len(data)+1)/2)-1],'centimeters .')
	error_counter = [error_counter[i]/error_max for i in range(len(error_counter))]
	plt.title(name+': CDF of Localization Error')
	new_ticks = np.linspace(0, 1.0, 11)
	plt.yticks(new_ticks)
	plt.ylabel('CDF')
	plt.xlabel('Error (cm)')
	plt.plot(range(int(range_num+1))[:500], error_counter[:500])
	plt.savefig(name+'_cdf.pdf')
	plt.show()
	return error_counter

def main(range_num=int):
	csv = load_data(file_name)
	norm_2_error = analysis_by_2_norm(csv)
	error = plt_cdf(file_name,norm_2_error,"DNN_FN RP7",range_num)

# 	csv = load_data(file_name1)
# 	norm_2_error = analysis_by_2_norm(csv)
# 	error1 = plt_cdf(file_name1,norm_2_error,"Multi-input",range_num)

# 	csv = load_data(file_name2)
# 	norm_2_error = analysis_by_2_norm(csv)
# 	error2 = plt_cdf(file_name2,norm_2_error,"Boosting",range_num)

# 	csv = load_data(file_name3)
# 	norm_2_error = analysis_by_2_norm(csv)
# 	error3 = plt_cdf(file_name3,norm_2_error,"SLN",range_num)

#	csv = load_data(file_name4)
#	norm_2_error = analysis_by_2_norm(csv)
#	error4 = plt_cdf(file_name4,norm_2_error,"SLN_FN",range_num)

# 	csv = load_data(file_name5)
# 	norm_2_error = analysis_by_2_norm(csv)
# 	error5 = plt_cdf(file_name5,norm_2_error,"XGBoost",range_num)

# 	csv = load_data(file_name6)
# 	norm_2_error = analysis_by_2_norm(csv)
# 	error6 = plt_cdf(file_name6,norm_2_error,"XGBoost_with time series data",range_num)

#	csv = load_data(file_name7)
#	norm_2_error = analysis_by_2_norm(csv)
#	error7 = plt_cdf(file_name7,norm_2_error,"XGBoost_FN",range_num)

	csv = load_data(file_name8)
	norm_2_error = analysis_by_2_norm(csv)
	error8 = plt_cdf(file_name8,norm_2_error,"DNN_FN",range_num)

#	csv = load_data(file_name9)
#	norm_2_error = analysis_by_2_norm(csv)
#	error9 = plt_cdf(file_name9,norm_2_error,"CNN2D",range_num)

#	csv = load_data(file_name10)
#	norm_2_error = analysis_by_2_norm(csv)
#	error10 = plt_cdf(file_name10,norm_2_error,"CNN2D_FN",range_num)

#	csv = load_data(file_name11)
#	norm_2_error = analysis_by_2_norm(csv)
#	error11 = plt_cdf(file_name11,norm_2_error,"CNN1D_FN",range_num)

#	csv = load_data(file_name12)
#	norm_2_error = analysis_by_2_norm(csv)
#	error12 = plt_cdf(file_name12,norm_2_error,"CNN1d_FN",range_num)

	plt.clf()
	plt.title('DNN+FN(RP7 VS RP6)')
	new_ticks = np.linspace(0, 1.0, 11)
	plt.yticks(new_ticks)
	plt.ylabel('CDF')
	plt.xlabel('Error (cm)')
	colors = ['r', 'c', 'm','lime', 'k','darkgray','aqua','darkorange','darksalmon','dodgerblue','indigo','lawngreen','cyan','gold']
	plt.plot(range(range_num+1), error[:(range_num+1)], c=colors[0])
#	plt.plot(range(range_num+1), error1[:(range_num+1)], c=colors[1])
#	plt.plot(range(range_num+1), error2[:(range_num+1)], c=colors[2])
#	plt.plot(range(range_num+1), error3[:(range_num+1)], c=colors[3])
#	plt.plot(range(range_num+1), error4[:(range_num+1)], c=colors[4])
#	plt.plot(range(range_num+1), error5[:(range_num+1)], c=colors[5])
#	plt.plot(range(range_num+1), error6[:(range_num+1)], c=colors[6])
#	plt.plot(range(range_num+1), error7[:(range_num+1)], c=colors[7])
	plt.plot(range(range_num+1), error8[:(range_num+1)], c=colors[8])
#	plt.plot(range(range_num+1), error9[:(range_num+1)], c=colors[9])
#	plt.plot(range(range_num+1), error10[:(range_num+1)], c=colors[10])
#	plt.plot(range(range_num+1), error11[:(range_num+1)], c=colors[11])
#	plt.plot(range(range_num+1), error12[:(range_num+1)], c=colors[12])

# 	plt.legend(['Fully connected model','Multi-input model','Boosting model','SLN model', 'SLN+FN model','XGBoost model','XGBoost model with time series data','XGBoost+FN model','DNN model','CNN model','DNN+FN model','CNN+FN model','CNN1d model'], loc='lower right')
	#plt.legend(['SLN_FN model','XGBoost_FN model','DNN_FN model','CNN2D model','CNN2D_FN','CNN1D_FN model'], loc='lower right')
	plt.legend(['DNN_FN 7 reference point','DNN_FN 6 reference point'], loc='lower right')
	plt.grid()
	plt.savefig('DNNRP6_VS_DNNRP7')
	plt.show()

if __name__ == "__main__":
	main(range_num)
