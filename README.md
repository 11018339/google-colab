**簡介**(README.md)
---------------------------------
Colaboratory (以下簡稱Colab)是一個在雲端運行的編輯執行環境，由Google提供開發者虛擬機，並支援Python程式及機器學習TensorFlow演算法。最棒的是只需要瀏覽器就可以運作，完全免費，目前官方建議使用Chrome，Firefox或Safari。

Colab目的在提供Machine Learning機器學習教育訓練及研究用，不須下載、不須安裝就可直接應用Python 2.7 與 Python 3.6資源庫，對初學者來說可以快速入門，不需耗時間在環境設定上。程式碼預設會直接儲存在開發者的Google Drive雲端硬碟中，執行時由虛擬機提供強大的運算能力，不會用到本機的資源。但要注意在閒置一段時間後，虛擬機會被停止並回收運算資源，此時只需再重新連接即可。接下來我們就來看操作環境：

**Colab Notebook環境介紹**

開啟chrome先登入google帳號，連結URL [https://colab.research.google.com/](https://colab.research.google.com/)，出現對話窗如下，
![image](https://github.com/11018339/google-colab/blob/main/images/1.jpg?raw=true)

按下右下角 NEW PYTHON 3 NOTEBOOK，出現如下的cell code區域。
![image](https://github.com/11018339/google-colab/blob/main/images/2.jpg?raw=true)

點擊 code cell進入編輯模式並貼上這段python程式碼：

import numpy
import matplotlib.pyplot as plt

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

按下左側執行button 或CTRL+Enter。會看到正態分布直方圖結果如下：
![image](https://github.com/11018339/google-colab/blob/main/images/3.jpg?raw=true)

對了，Colab有code IntelliSense功能，以上述範例來說，在前面兩行import完numpy等函式庫後，請先按下執行。接著再寫 x= numpy.random.n…編輯器會自動顯示代碼完成，參數信息，快速信息和成員列表等功能，十分方便，如下圖。
