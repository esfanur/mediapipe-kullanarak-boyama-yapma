import cv2
import  mediapipe as mp
import numpy as np

import  HandTrackingm as ht #burda elimizi agılamak ıcın yazdıgımız modulu buraya import edıyoruz
import  os #bunuda dosyadan gorsellerı cekmek ıcın kullıyoruz

folderPath="baslik"#burda klasor yolunu alıyoruz bızım resımlerımız baslık klasorunde
mylist=os.listdir(folderPath)
print(mylist) #dıyerek dosya ısımlerını okuyabıulırz
#ekranda gosterevcegmız resımlerı bır dızıye atalım
resimler=[]
drawcolor=(0,0,0)
for res in mylist:
    image=cv2.imread(f'{folderPath}/{res}')
    resimler.append(image)
print(len(resimler))

menu=resimler[0]
cam=cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)
#sımdı cızgımızı czdırmek ıcın baslangız noktalarına ıhtıyacımız car
cx,cy=0,0
#sımdı burada cızım yaptırmak ıcın canvas olusturuyoruz neden kameradan alınan yere cızemıyoruz ? bunun sebebı sureklı ekranımıza yenı goruntulerı alıp ekrana onun uzerıne eklmesınden dolayı whıle dongusunden kaynaklanıyor
#o yuzden bızde burda canvas olsuturup bu olusturdugumuz canvası ekranımıızın uzerıne bındırıerek  kamera ekranında kalmasını saglıycaz
canvas=np.zeros((720,1280,3),np.uint8) #uınt8 yanı bu 0 ile 255 arasında deger alacak
#elimizi algılayalım
detec=ht.handDetector(detectionCon=0.85,maxHands=1) #boyamıanın ıyo olması ıcın
tipsids=[4,8,12,16,20]#parmak uclarımızdakı landmark numaraları
while True:
    ret,frame=cam.read()
    frame=cv2.resize(frame,(1280,720))
    #resimizi ekrana yerlestırelım
    frame=cv2.flip(frame,1)

    #elın landmarklarını gosterelım
    frame=detec.findHands(frame)
    lmlist=detec.findPosition(frame,False) #burda yer ısaretlerının konumlarını etmek

    if len(lmlist)!=0: #yanı lmsit bos degılse konumları alıyorsa
        #burda lmlistden 8 ve 12 noktanın x ve y degerlerını almak ıstıyoruz
        x,y=lmlist[8][1:] #burda lmlist[8][1:] de yazılabılır
        x1,y1=lmlist[12][1:]

        fingers=detec.fingerup()
        print(fingers)
        #eger ıkı parmagımız yukardaysa
        if fingers[1] and fingers[2]:
            cx,cy=0,0 #burda sıfırlamamız gerekıyor  yoksa tekrar cızıme basladıgımız yerden son noktaya duz cızgı cekerrek baslar
            print("secim modunda")
            cv2.putText(frame,("Secim Modundasiniz"),(50,200),cv2.FONT_HERSHEY_COMPLEX,1,drawcolor,2)

            #simdi goruntunun en ustundeysek goruntumuzu veya secım modumuzu konuma gore desgırtırcez
            if y<125: #eger baslıktaysak
                if 250<x<450: #bırıncı resme tılıyoruz demektır
                    menu=resimler[0]
                    drawcolor=(255,0,0)
                elif 550<x<750: #ikiniv resme tılıyoruz demektır
                    menu=resimler[1]
                    drawcolor=(0,255,0)
                elif 800<x<950: #ucuncu resme tılıyoruz demektır
                    menu=resimler[2]
                    drawcolor=(0, 255, 255)
                elif 1050<x<1200: #bırıncı resme tılıyoruz demektır
                    menu=resimler[3]
                    drawcolor=(0,0,0)
            cv2.rectangle(frame, (x, y - 5), (x1, y1 + 5), drawcolor, cv2.FILLED)

        if fingers[1] and fingers[2]==False: #burda sadece ısaret parmagımızı kaldırıyoruz
            print("Çizim modunda")
            cv2.putText(frame, ("Cizim Modundasiniz"), (50, 200), cv2.FONT_HERSHEY_COMPLEX, 1, drawcolor, 2)
            cv2.circle(frame,(x,y),25,drawcolor,cv2.FILLED)
            if cx==0 and cy==0:
                cx,cy=x,y #burda baslangıc noktasını sıfır yapmak yerıne nerden cıızyorsak orayı alıyoruz
            cv2.line(frame,(cx,cy),(x,y),drawcolor,15)
            cv2.line(canvas, (cx, cy), (x, y), drawcolor, 15)
            cx,cy=x,y

    framegray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, framt = cv2.threshold(framegray, 60, 255,
                             cv2.THRESH_BINARY_INV)  # bunu kullanabılmek ıcın grı olması grekeıyordu o yuzden renk donusumu yaotık
    framt=cv2.cvtColor(framt,cv2.COLOR_GRAY2BGR)
    frame=cv2.bitwise_and(frame,framt)
    frame=cv2.bitwise_or(frame,canvas)
    #print(frame.shape,framt.shape) burda Görüntünün en boy oranı ve derinliği farklı olabilir onu kontrol ettık

    frame[0:120,0:1280]=menu
    #frame=cv2.addWeighted(frame,0.5,canvas,0.5,0)
    cv2.imshow("kamera",frame)
    #cv2.imshow("canvas",canvas)

    cv2.waitKey(1)

