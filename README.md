# GAN
GAN generative model 중에서 implicit density에 속하는 방법

<img src="./img/GAN_TAXONOMY.png" width="500" height="300">

GAN은 adversarial 구조로 image를 생성시키는 Generator와 이를 평가하는 Discriminator로 이루어지고 

이 둘의 관계는 서로 대립적인 구조로 성능을 점차 개선하는 방법이다.

![](img/adversarial2.PNG)

Generative model(G)는 실제 샘플X의 Distribution을 알아내려고 노력하고 

계속 노력하다보면  data distribution와 똑같은 분포를 만들어 Discriminator(D)가 실제와 가짜를 구별 할 수 없게됩니다.

결국에  D는 실수할 확률을 낮추기 위해서 노력하고(MAX) G는 실제 그림과 차이를 줄여서(MINI) 둘을 동시에 진행하므로  minimax problem 이라고 합니다.

![](img/minimax.PNG)

입장을 나누어서 생각하면 D는 구별을 잘하고 상황에서 실제 sample에 대해서는 D(x)=1 이여서 앞부분이 log(1)=0 이 되서 사라지고  
뒷부분에서는 G(z)가 이미지를 만들어낸 것이므로 D(G(z)=0 되도록 구별을 할 것이고 뒷부분도 log(1-0)=0이 되므로 이때가 D입장에서는 최댓값이라고 할 수 있습니다.

반대로 G의 입장에서는 뒷부분에 log(1-D(G(z)) 에서만 영향을 줄 수 있는데, G(z)를 실제 SAMPLE(X)와 같게 하여 x=G(z)가 되게 한다.
그래서 D(G(z))=1 이 되도록해서 log(1-D(G(z))을  최대한 MINIMIZE 시키기 합니다.

![](img/log.jpg)

하지만 논문에서는 다른 방법으로 G를 설정하는데, 학습이 초기에 진행 될 때 G(z)는 형편없는 이미지를 만들기 때문에 D는 구별을 아주  잘하게 됩니다.
그래서 BACKPROPAGATION을 할 때   log(1-D(G(z))의 Gradient는 아주 작은 값을 가지게 됩니다. 그래서 학습이 느려지게 되는데,
이 부분을 해결하기 위해 log(1-D(G(z))를 minimize하는것보다는 log(D(G(z))를 MAXIMIZE 하는 걸로 바꿔주게 되면 초기에도 강한 Gradient를  줄 수 있습니다.   

## 수식증명
![](img/수식1.PNG)

### OPTIMAL D 찾기
![](img/수식2.PNG)

### 최적일 때와  JS DIVERGENCE로 바꿔서 표현하기
![](img/수식3.PNG)


