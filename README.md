# defect_detection_for_edge_computing
現今製造業使用自動光學檢測(AOI)結合AI進行瑕疵檢測，然而AI龐大的運算使硬體裝置難以負荷，因此需要將模型部署至雲端進行推論，但此作法有網路延遲與資訊外洩的疑慮。另外，當工廠轉換產線或新增產品時，難以在短時間收集到足夠資料進行訓練。因此，我們提出一套「邊緣計算之遷移式銲錫瑕疵檢測系統」，先以微調遷移或基於Wasserstein距離之對抗式領域自適應使模型在目標資料集之準確率大幅提升，接著使用範數常態分佈離群值剪枝技術，搭配動態閾值設定，將模型之參數量降低以減少運算量，同時避免了使用少量資料訓練造成的過擬合現象，成功將微調遷移之模型在目標資料集的準確率，從原先的85%提升至90%，且推論速度提升55%，增加自動化工廠的生產效率。

# 檔案說明

### pruning_LiteonRacingData.py ###
針對使用光寶資料集訓練之ResNet模型進行剪枝，以下為使用者須自行修改之部分:  
`model_choose` : 模型選擇， 1 = ResNet50, 2 = ResNet101, 3 = ResNet152  
`criteria_choose` : 剪枝標準，1 = L1-norm, 2 = L2-norm, 3 = APoZ  
`regularization_choose` : 正則化方法，1 = LASSO, 2 = Ridge, 3 = Group_LASSO  
`path` : 模型辨識結果儲存路徑  
`weight_save_path` : 模型權重儲存路徑  
`model_save_path` : 模型儲存路徑  
`model_out_classnum` : 模型之輸出類別數量  

另也請修改函式`load_model()`中模型權重載入之路徑
***  


### pruning_standfordcar.py.py ###
針對使用公開資料集Stanford Cars Dataset訓練之模型進行剪枝，使用者須自行修改之部分同pruning_LiteonRacingData
***  


### wgan_trans_final.ipynb ###
以基於Wasserstein距離之對抗式領域自適應方法訓練模型，以此將源特徵域映射到目標域上
