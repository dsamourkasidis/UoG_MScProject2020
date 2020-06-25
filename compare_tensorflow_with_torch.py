import matplotlib.pyplot as plt
topk = [5,10,20]
#Clicks
clicks_Caser_hr_pytorch = [0.2793,0.3653,0.4443]
clicks_Caser_hr_tensorflow = [0.2728,0.3593,0.4371]
clicks_Caser_ndcg_pytorch = [0.1938, 0.2217, 0.2417]
clicks_Caser_ndcg_tensorflow = [0.1896,0.2177,0.2372]

#Purchases
purchases_Caser_hr_pytorch = [0.4499, 0.5689, 0.6565]
purchases_Caser_hr_tensorflow = [0.4475, 0.5559, 0.6393]
purchases_Caser_ndcg_pytorch = [0.3251, 0.3636, 0.3859]
purchases_Caser_ndcg_tensorflow = [0.3211,0.3565,0.3775]

plt.figure(1)
plt.plot(topk,clicks_Caser_hr_pytorch,'ro-',label = "Caser Pytorch")
plt.plot(topk,clicks_Caser_hr_tensorflow,'bo-',label = "Caser Tensorflow")
plt.title("HR score comparison for clicks")
plt.xlabel("Top-K values")
plt.ylabel("HR Score")
plt.legend()

plt.figure(2)
plt.plot(topk,clicks_Caser_ndcg_pytorch,'yo-',label = "Caser Pytorch")
plt.plot(topk,clicks_Caser_ndcg_tensorflow,'go-',label = "Caser Tensorflow")
plt.title("NDCG score comparison for clicks")
plt.xlabel("Top-K values")
plt.ylabel("NDCG Score")
plt.legend()

plt.figure(3)
plt.plot(topk,purchases_Caser_hr_pytorch,'ro-',label = "Caser Pytorch")
plt.plot(topk,purchases_Caser_hr_tensorflow,'bo-',label = "Caser Tensorflow")
plt.title("HR score comparison for purchases")
plt.xlabel("Top-K values")
plt.ylabel("HR Score")
plt.legend()

plt.figure(4)
plt.plot(topk,purchases_Caser_ndcg_pytorch,'yo-',label = "Caser Pytorch")
plt.plot(topk,purchases_Caser_ndcg_tensorflow,'go-',label = "Caser Tensorflow")
plt.title("NDCG score comparison for purchases")
plt.xlabel("Top-K values")
plt.ylabel("NDCG Score")
plt.legend()