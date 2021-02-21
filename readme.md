hackmd:
https://hackmd.io/@NTUTVOTT/Bk5gvsyzd
---
title: 'openCV tracker分給各個核心運算研究'
disqus: hackmd
---

openCV tracker 分給各個核心運算 PROCESS POOL 之研究(ubuntu18.04)
===

文件版本.：v0.0.1
[TOC]





## 1. 筆記解說

主要解決問題,如下圖,再多個追蹤也只再1個core上執行,例如下圖示追蹤3人 
也只用一個core來跑 

![](https://i.imgur.com/xLRdmUB.png)


主要是參考 參考文獻 [Multi-object tracking with dlib](https://www.pyimagesearch.com/2018/10/29/multi-object-tracking-with-dlib/) ,主要是透過MobileNetSSD將frame內人物框出(得出bounding box資訊),再利用**multiprocessing PROCESS POOL** 方法將frame與要追的目標等資訊利用各個邏輯核心生成process(非multi thread)去執行openCV tracker,規劃出兩種架構跟原始的 openCV multi-tracker 做出比較 

## 2. 各架構解說
### (1) 首先偵測出當前 frame 的 person 位置,會獲得 bounding box 資訊 
MobileNetSSD => 自動框出人物取得bounding box等資訊

已經從此程式分解出用法,可參考此 [github](https://github.com/masteree108/mobileNetSSD_test)
跑單張圖片執行結果
![](https://i.imgur.com/GWjqzoz.jpg)


### (2) 接下來的使用三種架構來對影片人物進行追蹤（ openCV tracker 都是使用 CSRT )

#### 方法1. 使用單一個邏輯核心執行
:::info
為並發運算 Concurrent Computing
:::
主要是使用 openCV 的 multi-tracker 方法
此方法只會在單一個邏輯核心上執行 tracking 任務,多人需追蹤時會依靠 multi-thread 來達成 
其作用為跟下面兩種架構進行比較 
補圖

#### 方法2.  使用多個 邏輯核心 執行
注意下方 邏輯核心-1的用意留1個邏輯核心給 OS 使用
:::info
為並行運算 parallel Computing
:::
##### 若偵測的人物 大於或等於 電腦的 邏輯核心-1 :
宣告大小為 電腦的  邏輯核心-1 數目的 process pool
讓 OS 分配 openCV tracker 在哪個 邏輯核心 上執行追蹤任務 
**(1個 邏輯核心 負責執行 *一個人物* 的追蹤)**
其他超出數目的追蹤任務必須等待指定的 邏輯核心 追蹤任務完成才會創立新的process進行追蹤

##### 若偵測的人物 小於 電腦的 邏輯核心-1:
宣告為該偵測人物數目的 process pool
讓 OS 分配 openCV tracker 在哪個 邏輯核心 上執行追蹤任務
補圖

#### 方法3. 使用多個 邏輯核心 + multi-thread 執行
:::info
為並行運算 parallel Computing + 並發運算 Concurrent Computing
:::
##### 若偵測的人物 大於或等於 電腦的 邏輯核心-1:
宣告大小為 電腦的 邏輯核心-1 數目的 process pool
讓 OS 分配 openCV multi tracker 在哪個 邏輯核心 上執行追蹤任務 
**(1個 邏輯核心 負責執行 *多個人物* 的追蹤)**
例如下圖追蹤人物為30人超出 邏輯核心:11 的數目,會安排下方的追蹤配置 
![](https://i.imgur.com/Tv71lbI.png)
前面的8個 邏輯核心 負責追蹤3人,後面的3個 邏輯核心 每個只需追蹤2人 

##### 若偵測的人物 小於 電腦的 邏輯核心-1:
宣告為該偵測人物數目的 process pool
讓OS分配 openCV multi-tracker 在哪個 邏輯核心 上執行追蹤任務
補圖


## 3. 執行結果 

#### 執行結果1 
顯示最快的為方法3 
其速度比原始的方法1平均快1.758秒 

![](https://i.imgur.com/DAdKeqT.png)

![](https://i.imgur.com/7R8diQo.png)


#### 執行結果2
顯示最快的為方法2 
其速度比原始的方法1平均快12.765秒 

![](https://i.imgur.com/FbQ36Sq.png)



![](https://i.imgur.com/jhcD6Zq.png)

#### 執行結果3
顯示最快的為方法3
其速度比原始的方法1平均快11.048秒 

![](https://i.imgur.com/w2Bt6ql.png)


![](https://i.imgur.com/f8DKg0I.png)

#### 執行結果4
顯示最快的為方法3
其速度比原始的方法1平均快21.559秒 
注意方法2在多個追蹤時速度+過大的影片size時 速度會至減慢跟方法1只差2秒 
![](https://i.imgur.com/m5SZmCW.png)


## 4. 範例程式與測試影片下載 
請至此 [github](https://github.com/masteree108/process_pool_MOT) 位置 下載 code 
```
$ git clone https://github.com/masteree108/process_pool_MOT.git
or 
$ git clone git@github.com:masteree108/process_pool_MOT.git
```

測試影片可至[此網址](https://drive.google.com/drive/folders/1UAoMQ07fZSa4XdDvKyOZhzxBrBSE1cxo?usp=sharing)下載 

## 5. 範例程式執行 
### 執行方法1的指令: 
若要改變觀察影片請修改如下檔案 
```
$ vim run_one_logical_core_cv_multi_tracker.sh
```
![](https://i.imgur.com/Ean0qbI.png)

若要改變frame size 請修改入下檔案
```
$ vim one_logical_core_cv_multi_tracker.py
```
![](https://i.imgur.com/DNFMG58.png)


執行指令
```
$ ./run_one_logical_core_cv_multi_tracker.sh
```


### 執行方法2的指令: 
若要改變觀察影片請修改如下檔案 
```
$ vim run_process_pool_cv_tracker.sh
```

若要改變frame size 請修改入下檔案
```
$ vim process_pool_cv_tracker.py
```

![](https://i.imgur.com/mIrTUCG.png)

執行指令
```
$ ./run_process_pool_cv_tracker.sh
```
### 執行方法3的指令: 
若要改變觀察影片請修改如下檔案 
```
$ vim run_process_pool_cv_multi_tracker.sh
```

若要改變frame size 請修改入下檔案
```
$ vim process_pool_cv_multi_tracker.py
```

![](https://i.imgur.com/CpnOCKM.png)


執行指令
```
$ ./run_process_pool_cv_multi_tracker.sh
```


##  6. 觀察 cpu loading 指令 ( ubuntu )

除了用top htop 等 command也能觀察之外,執行下列的指令 還能看到樹狀結構 
### 觀察方法2的指令: 
```
$ ps axu | grep [m]ulti_object_tracking_fast.py | awk '{print $2}' | xargs -n1 -I{} ps -o sid= -p {} | xargs -n1 -I{} ps --forest -o user,pid,ppid,cpuid,%cpu,%mem,stat,start,time,command -g {}

or 執行 script 如下 

$ ./watch_cpu_loading_PPCT.sh
```
### 觀察方法3的指令:
```
$ ps axu | grep [p]rocess_pool_cv_multi_tracker.py | awk '{print $2}' | xargs -n1 -I{} ps -o sid= -p {} | xargs -n1 -I{} ps --forest -o user,pid,ppid,cpuid,%c    pu,%mem,stat,start,time,command -g {}

or 執行 script 如下  

$ ./watch_cpu_loading_PPCMT.sh  
```

![](https://i.imgur.com/Za2rpIE.png)
294是所有core 再執行這process的使用率總和 




## 7. 其他指令 

cpu 可並行處理的數量公式 
![](https://i.imgur.com/oBlRyrO.png)
[這張圖的討論區](https://stackoverflow.com/questions/19225859/difference-between-core-and-processor)


```
Threads per core X cores per socket X sockets
```
![](https://i.imgur.com/prnS3go.png)

```
grep 'cpu cores' /proc/cpuinfo | uniq
```
![](https://i.imgur.com/KrsYtUx.png)

```
nproc --all
```
[參考網站](https://www.cyberciti.biz/faq/check-how-many-cpus-are-there-in-linux-system/)

## 8. 指定的processer重複問題
目前無解
![](https://i.imgur.com/HFKHVtP.png)





## 9. 參考文獻
[Multi-object tracking with dlib](https://www.pyimagesearch.com/2018/10/29/multi-object-tracking-with-dlib/)
[關於 multiprocessing module1](https://docs.python.org/3.4/library/multiprocessing.html#module-multiprocessing)
[關於 multiprocessing module2](https://sebastianraschka.com/Articles/2014_multiprocessing.html)

[A Hands on Guide to Multiprocessing in Python](https://towardsdatascience.com/a-hands-on-guide-to-multiprocessing-in-python-48b59bfcc89e)

[the pool class](https://sebastianraschka.com/Articles/2014_multiprocessing.html#the-pool-class)
![](https://i.imgur.com/xLOptsq.png)


[multiprocess 多進程組件Pool](https://www.itread01.com/content/1521659196.html)


[【Python教學】淺談 Multi-processing pool 使用方法](https://www.maxlist.xyz/2020/03/20/multi-processing-pool/)

[cv2.dnn.blobFromImage()函数用法](https://blog.csdn.net/weixin_42216109/article/details/103010206)

[CPU Cores VS Threads - Explained](youtube.com/watch?v=hwTYDQ0zZOw&feature=youtu.be)
![](https://i.imgur.com/JvM2qLd.png)
[
Difference between core and processor](https://stackoverflow.com/questions/19225859/difference-between-core-and-processor)
![](https://i.imgur.com/GT4W80Q.png)

[下圖分配給各個core與使用單一core的資料來源](https://www.geeksforgeeks.org/synchronization-pooling-processes-python/)
![](https://i.imgur.com/Vhydeco.png)

![](https://i.imgur.com/sj661Mi.png)





###### tags: `study`, `VoTT`
