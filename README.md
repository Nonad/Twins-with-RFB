# Twins-with-RFB

*upd:*

- [x] overfitting

- [x] output a number
- [x] output different numbers for different pics
- [ ] classification
- [ ] a *more correct* number

071221



https://github.com/Meituan-AutoML/Twins

https://github.com/dk-liang/TransCrowd

https://github.com/ruinmessi/RFBNet

So...

I managed to merge the model and some training files from all above with some *** I wrote to implement part of the idea of CCTrans(https://arxiv.org/abs/2109.14483)

_ (:з)∠) _

I am training it for a counting task

Yes, IT IS ABLE TO RUN in environment which satisfies requirements of TransCrowd （Don't forget to install NNI and take into account that the version of timm is older in Twins）

There is one of Twins' weight files (alt_gvt_base.pth, see Twins) used for my classification test (in folder  *cls*  )

It said connecting to **NNI** was timeout on my PC, that's why I comment some lines at the bottom of **temp.py**

Unfortunately, my recent experiment uncovered that it is easy to overfit on my less than a thousand pieces of images

Good luck if play with it! Welcome to discuss with me! (I mean, give me some suggestions plz)

Thanks for your attention (๑•̀ㅂ•́)و✧

NAD 181121

