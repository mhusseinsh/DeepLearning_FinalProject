import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

def plot_loss(history_train, history_train_scaled, lr_used, batches_used, best_mse, best_mse_scaled, network_mse_all, network_mse_scaled_all):
	fig1, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(history_train[best_mse])
	#axarr[0].plot(history_valid[best_mse])
	axarr[0].set_ylabel('loss')
	axarr[0].set_xlabel('epochs')
	axarr[0].set_title('Model Loss (raw data) ' + 'lr = ' + str(lr_used[best_mse]) + " ,batch = " + str(batches_used[best_mse]) + ' ,MSE = ' + str(network_mse_all[best_mse]))
	
	axarr[1].plot(history_train_scaled[best_mse_scaled])
	#axarr[1].plot(history_valid_scaled[best_mse_scaled])
	axarr[1].set_ylabel('loss')
	axarr[1].set_xlabel('epochs')
	axarr[1].set_title('Model Loss (scaled data) ' + 'lr = ' + str(lr_used[best_mse_scaled]) + " ,batch = " + str(batches_used[best_mse_scaled]) + ' ,MSE = ' + str(network_mse_scaled_all[best_mse_scaled]))
	fig1.set_size_inches(18.5, 10.5, forward=True)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig1.savefig('model_loss.png')

def plot_network_vs_true(targets, predictions, predictions_scaled, best_mse, best_mse_scaled, lr_used, batches_used):
	fig2, axarr = plt.subplots(2, sharex=True)
	axarr[0].scatter(targets, predictions, edgecolors=(0, 0, 0))
	axarr[0].plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=4)
	axarr[0].set_ylabel('Network Values')
	axarr[0].set_xlabel('True Values')
	axarr[0].set_title('True vs Network (raw data) ' + 'lr = ' + str(lr_used[best_mse]) + " ,batch = " + str(batches_used[best_mse]))
	
	axarr[1].scatter(targets, predictions_scaled, edgecolors=(0, 0, 0))
	axarr[1].plot([min(targets), max(targets)], [min(targets), max(targets)], 'k--', lw=4)
	axarr[1].set_ylabel('Network Values')
	axarr[1].set_xlabel('True Values')
	axarr[1].set_title('True vs Network (scaled data) ' + 'lr = ' + str(lr_used[best_mse_scaled]) + " ,batch = " + str(batches_used[best_mse_scaled]))
	fig2.set_size_inches(18.5, 10.5, forward=True)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig2.savefig('true_network.png')

def plot_learning_rate(l_rate, l_rate_scaled):
	fig3, axarr = plt.subplots(2, sharex=True)
	axarr[0].plot(l_rate)
	#axarr[0].plot(history_valid[best_mse])
	axarr[0].set_ylabel('learning rate')
	axarr[0].set_xlabel('epochs')
	axarr[0].set_title('Learning rate decay (raw data) ' + 'lr = ' + str(l_rate[0]))
	
	axarr[1].plot(l_rate_scaled)
	#axarr[1].plot(history_valid_scaled[best_mse_scaled])
	axarr[1].set_ylabel('learning rate')
	axarr[1].set_xlabel('epochs')
	axarr[1].set_title('Learning rate decay (scaled data) '  + 'lr = ' + str(l_rate_scaled[0]))
	fig1.set_size_inches(18.5, 10.5, forward=True)
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	fig1.savefig('learning_rate_decay.png')
	
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