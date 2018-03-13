import numpy as np
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def plot_loss_rnn(loss, params, select_time):
	fig1, ax = plt.subplots()
	ax.plot(loss)
	#axarr[0].plot(history_valid[best_mse])
	ax.set_ylabel('loss')
	ax.set_xlabel('epochs')
	ax.set_title('Model Loss with input length '+ str(select_time) + 
		', lr = ' + str(params[0]) + ", batch = " + str(params[1]) + ", alpha = " + str(params[4]))
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig1.savefig(str(select_time) +' rnn_loss.png')


def plot_predictions(predictions,targets,select_time):
	fig2, ax = plt.subplots()
	ax.plot(predictions)
	ax.plot(targets)
	#axarr[0].plot(history_valid[best_mse])
	ax.set_ylabel('values')
	ax.set_xlabel('epochs')
	#ax.legend(['predictions', 'true'], loc='best', fancybox=True, framealpha=0.5)
	ax.set_title('Predictions vs True with time length'+ str(select_time))
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig2.savefig(str(select_time) +' rnn_true_prediction_plot.png')

def plot_loss(raw_loss,scaled_loss ,params_raw,params_scaled, mse_raw, mse_scaled):
	fig3, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(raw_loss)
	#axarr[0].plot(history_valid[best_mse])
	axarr[0].set_ylabel('loss')
	axarr[0].set_xlabel('epochs')
	axarr[0].set_title('Model Loss (raw data): MSE = ' + str(mse_raw) + ', lr = ' + str(params_raw[0]) + ", batch = " + str(params_raw[2]) + ', alpha = ' + str(params_raw[1]))
	
	axarr[1].plot(scaled_loss)
	#axarr[1].plot(history_valid_scaled[best_mse_scaled])
	axarr[1].set_ylabel('loss')
	axarr[1].set_xlabel('epochs')
	axarr[1].set_title('Model Loss (scaled data): MSE = ' + str(mse_scaled) + ', lr = ' + str(params_scaled[0]) + ", batch = " + str(params_scaled[2]) + ', alpha = ' + str(params_scaled[1]))
	fig3.set_size_inches(18.5, 10.5, forward=True)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig3.savefig('model_loss.png')


def plot_network_vs_true(targets, predictions, predictions_scaled, params_raw, params_scaled, mse_raw, mse_scaled):
	fig4, axarr = plt.subplots(2, sharex=True)
	axarr[0].scatter(targets, predictions, edgecolors=(0, 0, 0))
	axarr[0].plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=4)
	axarr[0].set_ylabel('Network Values')
	axarr[0].set_xlabel('True Values')
	axarr[0].set_title('True vs Network (raw data): MSE = ' + str(mse_raw) + ', lr = ' + str(params_raw[0]) + " ,batch = " + str(params_raw[2]) + ', alpha = ' + str(params_raw[1]))
	
	axarr[1].scatter(targets, predictions_scaled, edgecolors=(0, 0, 0))
	axarr[1].plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=4)
	axarr[1].set_ylabel('Network Values')
	axarr[1].set_xlabel('True Values')
	axarr[1].set_title('True vs Network (scaled data): MSE = ' + str(mse_scaled) + ', lr = '+ str(params_scaled[0]) + " ,batch = " + str(params_scaled[2]) + ', alpha = ' + str(params_scaled[1]))
	fig4.set_size_inches(18.5, 10.5, forward=True)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig4.savefig('true_network.png')

def plot_rnn_vs_true(targets, predictions, params, select_time):
	fig5, ax = plt.subplots()
	ax.scatter(targets, predictions, edgecolors=(0, 0, 0))
	ax.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=4)
	ax.set_ylabel('RNN Values')
	ax.set_xlabel('True Values')
	ax.set_title('True vs RNN with input length '+ str(select_time) + ', lr = ' + str(params[0]) + " ,batch = " + str(params[1])
		 + ", alpha = " + str(params[4]))
	
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig5.savefig(str(select_time) +' true_rnn.png')

def plot_baseline_vs_true(targets, baseline_predictions, mse):
	fig6, ax = plt.subplots()
	ax.scatter(targets, baseline_predictions, edgecolors=(0, 0, 0))
	ax.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=4)
	ax.set_ylabel('Baseline Values')
	ax.set_xlabel('True Values')
	ax.set_title('True vs Baseline MSE = ' + str(mse))
	fig6.savefig('true_baseline.png')

def plot_baseline_vs_true2(targets, baseline_predictions, mse):
	fig6, ax = plt.subplots()
	ax.scatter(targets, baseline_predictions, edgecolors=(0, 0, 0))
	ax.plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=4)
	ax.set_ylabel('Baseline Values')
	ax.set_xlabel('True Values')
	ax.set_title('True vs Baseline MSE = ' + str(mse))
	fig6.savefig('2true_baseline.png')

#def plot_learning_curves(all_predictions, targets, params, select_time,mse_all):
def plot_learning_curves(max_pred, max_t, min_pred, min_t, params, select_time,mse_max, mse_min,time):
	#transposed = np.array(all_predictions).T

	fig7, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(max_pred)
	axarr[0].plot(max_t)

	axarr[0].set_ylabel('values')
	axarr[0].set_xlabel('epoch')
	axarr[0].set_title('Predictions with input length '+ str(select_time) + 'lr = ' + str(params[0]) + " ,batch = " + str(params[1])
			 + ", alpha = " + str(params[4]) + ", MSE= "+str(mse_max), fontsize=20, fontweight="bold")
	
	axarr[1].plot(min_pred)
	axarr[1].plot(min_t)

	axarr[1].set_ylabel('values')
	axarr[1].set_xlabel('epoch')
	axarr[1].set_title('Predictions with input length '+ str(select_time) + 'lr = ' + str(params[0]) + " ,batch = " + str(params[1])
			 + ", alpha = " + str(params[4]) + ", MSE= "+str(mse_min), fontsize=20, fontweight="bold")
	fig7.set_size_inches(18.5, 10.5, forward=True)
	plt.legend(['prediction', 'true'], loc='best', fancybox=True, framealpha=0.5)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig7.savefig('Model '+str(time) + ' with data '+str(select_time) + '_best_and_worst_predictions_true_curves.png')
	
def plot_learning_curves_random(max_pred, max_t, min_pred, min_t, params, select_time,mse_max, mse_min):
	#transposed = np.array(all_predictions).T

	fig8, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(max_pred)
	axarr[0].plot(max_t)

	axarr[0].set_ylabel('values')
	axarr[0].set_xlabel('epoch')
	axarr[0].set_title('Predictions with input length '+ str(select_time) + 'lr = ' + str(params[0]) 
			 + ", alpha = " + str(params[2]) + ", MSE= "+str(mse_max), fontsize=20, fontweight="bold")
	
	axarr[1].plot(min_pred)
	axarr[1].plot(min_t)

	axarr[1].set_ylabel('values')
	axarr[1].set_xlabel('epoch')
	axarr[1].set_title('Predictions with input length '+ str(select_time) + 'lr = ' + str(params[0]) 
			 + ", alpha = " + str(params[2]) + ", MSE= "+str(mse_min), fontsize=20, fontweight="bold")
	fig8.set_size_inches(18.5, 10.5, forward=True)
	plt.legend(['prediction', 'true'], loc='best', fancybox=True, framealpha=0.5)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig8.savefig('Randomized_'+str(select_time) + '_best_and_worst_predictions_true_curves.png')

def plot_all_learning_curves(predictions, targets, params, train_time, pred_time, mses):
	print("Begin Plotting")
	for col in range(len(predictions)):
		fig9, ax = plt.subplots()
		ax.plot(predictions[col])
		ax.plot(targets[col])
		plt.axvline(x=pred_time-1, linestyle='dashed', linewidth=2.0, color = 'black')
		ax.set_title('Predictions with input length '+ str(train_time) + ' and pred time: '+ str(pred_time) + ' lr = ' + str(params[0]) 
			 + ", alpha = " + str(params[1]) + ", MSE = "+str(mses), fontsize=20, fontweight="bold")
		ax.set_ylabel('values')
		ax.set_xlabel('epoch')
		fig9.set_size_inches(18.5, 10.5, forward=True)
		plt.legend(['prediction', 'true'], loc='best', fancybox=True, framealpha=0.5)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig9.savefig('./Plots/Train ' + str(train_time) + '/Test ' + str(pred_time) + '/' + str(col)+ '_' + str(train_time) + '_' + str(pred_time)+ '.png')
		plt.close()

def plot_box_plots(predictions, params, train_time, pred_time):

		figg, ax = plt.subplots()
		#for i in (predictions):
		plt.boxplot([np.log(predictions[0]), np.log(predictions[1]), np.log(predictions[2]), np.log(predictions[3])], showmeans=True, meanline=False)#, labels=([str(pred_time) + ' epochs'])
		ax.set_xticklabels(['5 epochs', '10 epochs', '20 epochs', '30 epochs'])
		plt.xlabel("Test")
		plt.ylabel("log(MSE)")
		plt.title("log (MSE) Quartile", fontsize=20, fontweight="bold")
		plt.tight_layout()
		figg.set_size_inches(18.5, 10.5, forward=True)
		plt.subplots_adjust(top=0.85)
		figg.savefig('./Plots/Train ' + str(train_time) + '/MSE (Boxplot).png')
		plt.close()

def plot(dimension):
		# Loss (raw data)
		dimension = int(math.sqrt(dimension))
		fig1, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.plot(history[cnt].history['loss'])
				col.plot(history[cnt].history['val_loss'])
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]))
				col.set_ylabel('loss')
				col.set_xlabel('epoch')
				cnt+=1
				col.legend(['train', 'test'], loc='best', fancybox=True, framealpha=0.5)
		plt.suptitle('Model Loss (raw data)', fontsize=20, fontweight="bold")
		fig1.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig1.savefig('model_loss_raw.png')
		
		# Loss (scaled data)
		fig2, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.plot(history_scaled[cnt].history['loss'])
				col.plot(history_scaled[cnt].history['val_loss'])
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]))
				col.set_ylabel('loss')
				col.set_xlabel('epoch')
				cnt+=1
				col.legend(['train', 'test'], loc='best', fancybox=True, framealpha=0.5)
		plt.suptitle('Model Loss (scaled data)', fontsize=20, fontweight="bold")
		fig2.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig2.savefig('model_loss_scaled.png')

		# True vs Baseline (raw data)
		fig3, ax = plt.subplots()
		ax.scatter(y_sorted, y_pred[0], edgecolors=(0, 0, 0))
		ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
		ax.set_xlabel('True Values')
		ax.set_ylabel('Baseline Values')
		ax.set_title('True vs Baseline (raw data)', fontsize=20, fontweight="bold")
		#plt.show()
		plt.savefig('model_rawData(baseline).png')

		#True vs Network (raw data)
		fig4, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.scatter(y_sorted, y_net[cnt], edgecolors=(0, 0, 0))
				col.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all[cnt]))
				col.set_ylabel('Network Values')
				col.set_xlabel('True Values')
				cnt+=1
		plt.suptitle('True vs Network (raw data)', fontsize=20, fontweight="bold")
		fig4.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig4.savefig('model_rawData(network).png')

		# True vs Baseline vs Network (raw)
		fig5, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.plot(y_sorted)
				col.plot(y_pred[cnt])
				col.plot(y_net[cnt])
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all[cnt]))
				col.set_ylabel('y Value')
				col.set_xlabel('Samples')
				col.legend(['true', 'baseline', 'network'], loc='best', fancybox=True, framealpha=0.5)
				cnt+=1
		plt.suptitle('Comparison (raw data)', fontsize=20, fontweight="bold")
		fig5.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig5.savefig('metrics_comparison_raw.png')

		# True vs Baseline vs Network (scaled)
		fig6, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.plot(y_sorted)
				col.plot(y_pred_scaled[cnt])
				col.plot(y_net_scaled[cnt])
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all_scaled[cnt]))
				col.set_ylabel('y Value')
				col.set_xlabel('Samples')
				col.legend(['true', 'baseline', 'network'], loc='best', fancybox=True, framealpha=0.5)
				cnt+=1
		plt.suptitle('Comparison (scaled data)', fontsize=20, fontweight="bold")
		fig6.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig6.savefig('metrics_comparison_scaled.png')

		# True vs Baseline (scaled data)
		fig7, ax = plt.subplots()
		ax.scatter(y_sorted, y_pred_scaled[0], edgecolors=(0, 0, 0))
		ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
		ax.set_xlabel('True Values')
		ax.set_ylabel('Baseline Values')
		ax.set_title('True vs Baseline (scaled data)', fontsize=20, fontweight="bold")
		#plt.show()
		plt.savefig('model_scaledData(baseline).png')

		#True vs Network (scaled data)
		fig8, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.scatter(y_sorted, y_net_scaled[cnt], edgecolors=(0, 0, 0))
				col.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all_scaled[cnt]))
				col.set_ylabel('Network Values')
				col.set_xlabel('True Values')
				cnt+=1
		plt.suptitle('True vs Network (scaled data)', fontsize=20, fontweight="bold")
		fig8.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig8.savefig('model_scaledData(network).png')

"""		# LAST 4 LABELS PLOTTING

		# True vs Baseline (raw data)
		fig9, ax = plt.subplots()
		ax.scatter(y_sorted, y_pred_4[0], edgecolors=(0, 0, 0))
		ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
		ax.set_xlabel('True Values')
		ax.set_ylabel('Baseline Values')
		ax.set_title('Last 4 - True vs Baseline (raw data)', fontsize=20, fontweight="bold")
		#plt.show()
		plt.savefig('model_rawData_4(baseline).png')

		#True vs Network (raw data)
		fig10, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.scatter(y_sorted, y_net_4[cnt], edgecolors=(0, 0, 0))
				col.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all_4[cnt]))
				col.set_ylabel('Network Values')
				col.set_xlabel('True Values')
				cnt+=1
		plt.suptitle('Last 4 - True vs Network (raw data)', fontsize=20, fontweight="bold")
		fig4.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig4.savefig('model_rawData_4(network).png')

		# True vs Baseline vs Network (raw)
		fig11, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.plot(y_sorted)
				col.plot(y_pred_4[cnt])
				col.plot(y_net_4[cnt])
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all_4[cnt]))
				col.set_ylabel('y Value')
				col.set_xlabel('Samples')
				col.legend(['true', 'baseline', 'network'], loc='best', fancybox=True, framealpha=0.5)
				cnt+=1
		plt.suptitle('Last 4 - Comparison (raw data)', fontsize=20, fontweight="bold")
		fig5.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig5.savefig('metrics_comparison_raw_4.png')

		# True vs Baseline vs Network (scaled)
		fig12, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.plot(y_sorted)
				col.plot(y_pred_scaled_4[cnt])
				col.plot(y_net_scaled_4[cnt])
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all_scaled_4[cnt]))
				col.set_ylabel('y Value')
				col.set_xlabel('Samples')
				col.legend(['true', 'baseline', 'network'], loc='best', fancybox=True, framealpha=0.5)
				cnt+=1
		plt.suptitle('Last 4 - Comparison (scaled data)', fontsize=20, fontweight="bold")
		fig6.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig6.savefig('metrics_comparison_scaled_4.png')

		# True vs Baseline (scaled data)
		fig13, ax = plt.subplots()
		ax.scatter(y_sorted, y_pred_scaled_4[0], edgecolors=(0, 0, 0))
		ax.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
		ax.set_xlabel('True Values')
		ax.set_ylabel('Baseline Values')
		ax.set_title('Last 4 - True vs Baseline (scaled data)', fontsize=20, fontweight="bold")
		#plt.show()
		plt.savefig('model_scaledData(baseline)_4.png')

		#True vs Network (scaled data)
		fig14, ax = plt.subplots(dimension, dimension)
		cnt = 0
		for row in ax:
			for col in row:
				col.scatter(y_sorted, y_net_scaled_4[cnt], edgecolors=(0, 0, 0))
				col.plot([min(y_sorted), max(y_sorted)], [min(y_sorted), max(y_sorted)], 'k--', lw=4)
				col.set_title('lr =' + str(lr_used[cnt]) + " ,batch =" + str(batches_used[cnt]) + ' ,MSE =' + str(mse_all_scaled_4[cnt]))
				col.set_ylabel('Network Values')
				col.set_xlabel('True Values')
				cnt+=1
		plt.suptitle('Last 4 - True vs Network (scaled data)', fontsize=20, fontweight="bold")
		fig8.set_size_inches(18.5, 10.5, forward=True)
		plt.tight_layout()
		plt.subplots_adjust(top=0.85)
		fig8.savefig('model_scaledData(network)_4.png')"""