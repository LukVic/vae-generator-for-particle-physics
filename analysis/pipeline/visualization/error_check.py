import matplotlib.pyplot as plt
import numpy as np
import ROOT
import array

def LoadStyle():
    ROOT.gStyle.SetPadLeftMargin(0.15)
    ROOT.gStyle.SetPadBottomMargin(0.15)
    ROOT.gStyle.SetPadTopMargin(0.05)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetEndErrorSize(0.0)
    ROOT.gStyle.SetTitleSize(0.05,"X")
    ROOT.gStyle.SetTitleSize(0.045,"Y")
    ROOT.gStyle.SetLabelSize(0.045,"X")
    ROOT.gStyle.SetLabelSize(0.045,"Y")
    ROOT.gStyle.SetTitleOffset(1.2,"X")
    ROOT.gStyle.SetTitleOffset(1.35,"Y")
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetHatchesSpacing(0.3)

seed_32_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.99018 | PREC TE: 0.88402 | ACC TR: 0.99071 | ACC TE: 0.84439 | SIGT: 11.00567 | SIGS: 14.25860 | BS: 27.88054 | BB: 5.17289 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93567 | PREC TE: 0.89391 | ACC TR: 0.91633 | ACC TE: 0.85769 | SIGT: 11.91860 | SIGS: 18.05622 | BS: 23.41021 | BB: 1.90809 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92549 | PREC TE: 0.89678 | ACC TR: 0.90284 | ACC TE: 0.86262 | SIGT: 12.06955 | SIGS: 17.80640 | BS: 27.18613 | BB: 3.05420 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92096 | PREC TE: 0.89835 | ACC TR: 0.89505 | ACC TE: 0.86245 | SIGT: 12.04150 | SIGS: 17.46748 | BS: 27.31681 | BB: 3.14211 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91827 | PREC TE: 0.89757 | ACC TR: 0.89098 | ACC TE: 0.86324 | SIGT: 12.26152 | SIGS: 18.69364 | BS: 26.77751 | BB: 2.67306 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91786 | PREC TE: 0.89791 | ACC TR: 0.89015 | ACC TE: 0.86259 | SIGT: 12.33358 | SIGS: 19.64685 | BS: 23.93554 | BB: 1.72195 | PARAMS: 0',
]
seed_56_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.97958 | PREC TE: 0.88443 | ACC TR: 0.97917 | ACC TE: 0.84873 | SIGT: 12.12346 | SIGS: 16.74949 | BS: 25.75087 | BB: 2.46421 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93478 | PREC TE: 0.89324 | ACC TR: 0.91523 | ACC TE: 0.85835 | SIGT: 12.51527 | SIGS: 18.39324 | BS: 26.65133 | BB: 2.36369 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92606 | PREC TE: 0.89504 | ACC TR: 0.90263 | ACC TE: 0.85999 | SIGT: 12.63764 | SIGS: 18.48887 | BS: 28.23968 | BB: 2.76703 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92002 | PREC TE: 0.89822 | ACC TR: 0.89394 | ACC TE: 0.86229 | SIGT: 12.55549 | SIGS: 18.80454 | BS: 27.09039 | BB: 2.46573 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91872 | PREC TE: 0.89873 | ACC TR: 0.89067 | ACC TE: 0.86341 | SIGT: 12.65939 | SIGS: 18.78965 | BS: 30.20699 | BB: 3.47880 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91630 | PREC TE: 0.89855 | ACC TR: 0.88929 | ACC TE: 0.86495 | SIGT: 12.75199 | SIGS: 18.72880 | BS: 30.06505 | BB: 3.29941 | PARAMS: 0',
]
seed_17_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98457 | PREC TE: 0.88084 | ACC TR: 0.98288 | ACC TE: 0.85083 | SIGT: 11.26621 | SIGS: 14.25673 | BS: 32.86198 | BB: 7.94697 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93560 | PREC TE: 0.89386 | ACC TR: 0.91653 | ACC TE: 0.85930 | SIGT: 11.96560 | SIGS: 16.77570 | BS: 28.26835 | BB: 3.64972 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92212 | PREC TE: 0.89516 | ACC TR: 0.89848 | ACC TE: 0.86255 | SIGT: 12.11954 | SIGS: 16.83606 | BS: 29.09078 | BB: 3.78450 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91978 | PREC TE: 0.89811 | ACC TR: 0.89338 | ACC TE: 0.86534 | SIGT: 12.33887 | SIGS: 18.42506 | BS: 27.76994 | BB: 2.94525 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91795 | PREC TE: 0.89823 | ACC TR: 0.89147 | ACC TE: 0.86571 | SIGT: 12.23775 | SIGS: 17.98935 | BS: 25.65978 | BB: 2.31738 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91585 | PREC TE: 0.89859 | ACC TR: 0.88844 | ACC TE: 0.86600 | SIGT: 12.41466 | SIGS: 17.95119 | BS: 25.65937 | BB: 2.14877 | PARAMS: 0',
]
seed_98_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98962 | PREC TE: 0.88497 | ACC TR: 0.98846 | ACC TE: 0.84853 | SIGT: 12.06866 | SIGS: 15.99073 | BS: 26.82085 | BB: 2.91367 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93295 | PREC TE: 0.89780 | ACC TR: 0.91302 | ACC TE: 0.86285 | SIGT: 12.33891 | SIGS: 18.99577 | BS: 27.14220 | BB: 2.71602 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92573 | PREC TE: 0.89858 | ACC TR: 0.90108 | ACC TE: 0.86364 | SIGT: 12.62733 | SIGS: 18.59131 | BS: 26.37881 | BB: 2.17227 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92028 | PREC TE: 0.89965 | ACC TR: 0.89502 | ACC TE: 0.86630 | SIGT: 12.80221 | SIGS: 19.10078 | BS: 27.29820 | BB: 2.28862 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91921 | PREC TE: 0.89994 | ACC TR: 0.89236 | ACC TE: 0.86607 | SIGT: 12.74419 | SIGS: 19.96149 | BS: 28.20821 | BB: 2.75334 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91649 | PREC TE: 0.89914 | ACC TR: 0.88923 | ACC TE: 0.86590 | SIGT: 12.81996 | SIGS: 20.65236 | BS: 30.28496 | BB: 3.29571 | PARAMS: 0',
]
seed_42_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.97821 | PREC TE: 0.88481 | ACC TR: 0.97386 | ACC TE: 0.84902 | SIGT: 11.70272 | SIGS: 15.54303 | BS: 30.21123 | BB: 5.08011 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93327 | PREC TE: 0.89392 | ACC TR: 0.91344 | ACC TE: 0.86167 | SIGT: 12.58083 | SIGS: 19.19226 | BS: 25.61897 | BB: 1.98993 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92398 | PREC TE: 0.89821 | ACC TR: 0.89915 | ACC TE: 0.86354 | SIGT: 12.41319 | SIGS: 17.36892 | BS: 30.12148 | BB: 3.79556 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91926 | PREC TE: 0.89798 | ACC TR: 0.89247 | ACC TE: 0.86469 | SIGT: 12.54779 | SIGS: 17.17098 | BS: 30.71256 | BB: 3.84985 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91900 | PREC TE: 0.90046 | ACC TR: 0.89160 | ACC TE: 0.86804 | SIGT: 12.41376 | SIGS: 18.79902 | BS: 26.11751 | BB: 2.29250 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91697 | PREC TE: 0.90043 | ACC TR: 0.88915 | ACC TE: 0.86728 | SIGT: 12.61936 | SIGS: 19.60990 | BS: 30.18313 | BB: 3.52431 | PARAMS: 0',
]
seed_73_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98801 | PREC TE: 0.88532 | ACC TR: 0.98779 | ACC TE: 0.84833 | SIGT: 11.86551 | SIGS: 15.81614 | BS: 25.28103 | BB: 2.57676 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93470 | PREC TE: 0.89482 | ACC TR: 0.91615 | ACC TE: 0.85917 | SIGT: 12.34404 | SIGS: 19.06700 | BS: 28.35313 | BB: 3.16191 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92532 | PREC TE: 0.89584 | ACC TR: 0.90069 | ACC TE: 0.86114 | SIGT: 12.64033 | SIGS: 18.83817 | BS: 28.50465 | BB: 2.96129 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92007 | PREC TE: 0.89752 | ACC TR: 0.89414 | ACC TE: 0.86311 | SIGT: 12.57968 | SIGS: 19.09221 | BS: 30.29040 | BB: 3.62418 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91814 | PREC TE: 0.89999 | ACC TR: 0.89149 | ACC TE: 0.86722 | SIGT: 12.62559 | SIGS: 20.78671 | BS: 28.31470 | BB: 2.82859 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91663 | PREC TE: 0.89953 | ACC TR: 0.88884 | ACC TE: 0.86640 | SIGT: 12.70096 | SIGS: 20.45049 | BS: 26.52723 | BB: 2.14976 | PARAMS: 0',
]
seed_5_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98836 | PREC TE: 0.88745 | ACC TR: 0.98673 | ACC TE: 0.84935 | SIGT: 11.55443 | SIGS: 14.33344 | BS: 29.23502 | BB: 4.83244 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93300 | PREC TE: 0.89332 | ACC TR: 0.91313 | ACC TE: 0.85743 | SIGT: 11.87490 | SIGS: 17.75600 | BS: 26.21721 | BB: 2.96105 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92468 | PREC TE: 0.89619 | ACC TR: 0.90092 | ACC TE: 0.86242 | SIGT: 12.00845 | SIGS: 16.90937 | BS: 27.01910 | BB: 3.06583 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92117 | PREC TE: 0.89912 | ACC TR: 0.89470 | ACC TE: 0.86423 | SIGT: 12.21038 | SIGS: 17.17708 | BS: 26.17037 | BB: 2.51647 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91882 | PREC TE: 0.89935 | ACC TR: 0.89122 | ACC TE: 0.86410 | SIGT: 12.07643 | SIGS: 17.33640 | BS: 27.61053 | BB: 3.21486 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91653 | PREC TE: 0.89819 | ACC TR: 0.88895 | ACC TE: 0.86462 | SIGT: 12.18390 | SIGS: 17.88366 | BS: 29.08897 | BB: 3.68713 | PARAMS: 0',
]
seed_61_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98785 | PREC TE: 0.87932 | ACC TR: 0.98819 | ACC TE: 0.84623 | SIGT: 11.88494 | SIGS: 17.27943 | BS: 25.74692 | BB: 2.72604 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93729 | PREC TE: 0.89432 | ACC TR: 0.91710 | ACC TE: 0.85933 | SIGT: 12.37879 | SIGS: 18.46876 | BS: 29.35580 | BB: 3.51978 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92566 | PREC TE: 0.89432 | ACC TR: 0.90224 | ACC TE: 0.86130 | SIGT: 12.61240 | SIGS: 18.53385 | BS: 29.49802 | BB: 3.26144 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92041 | PREC TE: 0.89589 | ACC TR: 0.89526 | ACC TE: 0.86278 | SIGT: 12.33071 | SIGS: 19.62890 | BS: 28.07034 | BB: 3.06922 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91909 | PREC TE: 0.89816 | ACC TR: 0.89223 | ACC TE: 0.86544 | SIGT: 12.60217 | SIGS: 19.69315 | BS: 29.09173 | BB: 3.11891 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91641 | PREC TE: 0.89617 | ACC TR: 0.88937 | ACC TE: 0.86278 | SIGT: 12.52816 | SIGS: 19.61246 | BS: 29.45099 | BB: 3.35387 | PARAMS: 0',
]
seed_89_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98927 | PREC TE: 0.88613 | ACC TR: 0.98859 | ACC TE: 0.84633 | SIGT: 11.30920 | SIGS: 14.75835 | BS: 24.65780 | BB: 3.00311 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93602 | PREC TE: 0.89431 | ACC TR: 0.91876 | ACC TE: 0.85855 | SIGT: 11.70278 | SIGS: 16.91980 | BS: 28.40432 | BB: 4.12960 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92493 | PREC TE: 0.89551 | ACC TR: 0.90263 | ACC TE: 0.86107 | SIGT: 12.11037 | SIGS: 16.89704 | BS: 29.12310 | BB: 3.81297 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92020 | PREC TE: 0.89539 | ACC TR: 0.89443 | ACC TE: 0.86022 | SIGT: 12.08788 | SIGS: 16.72348 | BS: 27.60009 | BB: 3.19553 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91808 | PREC TE: 0.89830 | ACC TR: 0.89142 | ACC TE: 0.86259 | SIGT: 12.10601 | SIGS: 18.55429 | BS: 27.64302 | BB: 3.18903 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91723 | PREC TE: 0.89745 | ACC TR: 0.88927 | ACC TE: 0.86262 | SIGT: 12.34630 | SIGS: 18.45244 | BS: 24.85906 | BB: 1.97032 | PARAMS: 0',
]
seed_25_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98406 | PREC TE: 0.88459 | ACC TR: 0.98394 | ACC TE: 0.84846 | SIGT: 11.87059 | SIGS: 14.92741 | BS: 27.53232 | BB: 3.46517 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93470 | PREC TE: 0.89437 | ACC TR: 0.91466 | ACC TE: 0.85920 | SIGT: 12.34680 | SIGS: 18.95761 | BS: 28.85865 | BB: 3.35960 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92492 | PREC TE: 0.89697 | ACC TR: 0.90077 | ACC TE: 0.86268 | SIGT: 12.42612 | SIGS: 18.41326 | BS: 27.01568 | BB: 2.57614 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92041 | PREC TE: 0.89815 | ACC TR: 0.89494 | ACC TE: 0.86406 | SIGT: 12.26587 | SIGS: 18.18910 | BS: 28.22946 | BB: 3.21440 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91775 | PREC TE: 0.89733 | ACC TR: 0.89144 | ACC TE: 0.86452 | SIGT: 12.65842 | SIGS: 18.04003 | BS: 29.15425 | BB: 3.07265 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91650 | PREC TE: 0.89943 | ACC TR: 0.88954 | ACC TE: 0.86702 | SIGT: 12.42579 | SIGS: 18.52684 | BS: 28.88236 | BB: 3.26386 | PARAMS: 0',
]
seed_10_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.99071 | PREC TE: 0.88501 | ACC TR: 0.98885 | ACC TE: 0.84876 | SIGT: 12.22842 | SIGS: 16.22944 | BS: 32.07406 | BB: 5.05030 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93550 | PREC TE: 0.89668 | ACC TR: 0.91655 | ACC TE: 0.86160 | SIGT: 12.55219 | SIGS: 17.94816 | BS: 30.04516 | BB: 3.56160 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92608 | PREC TE: 0.89648 | ACC TR: 0.90356 | ACC TE: 0.86222 | SIGT: 12.71200 | SIGS: 18.63178 | BS: 30.28205 | BB: 3.43728 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92147 | PREC TE: 0.89834 | ACC TR: 0.89614 | ACC TE: 0.86291 | SIGT: 12.67186 | SIGS: 18.29679 | BS: 30.92130 | BB: 3.75499 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91781 | PREC TE: 0.89749 | ACC TR: 0.89159 | ACC TE: 0.86367 | SIGT: 12.89194 | SIGS: 18.08601 | BS: 31.45775 | BB: 3.66059 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91658 | PREC TE: 0.89847 | ACC TR: 0.88971 | ACC TE: 0.86492 | SIGT: 12.84476 | SIGS: 20.24432 | BS: 29.61814 | BB: 3.01676 | PARAMS: 0',
]
seed_3_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98461 | PREC TE: 0.88735 | ACC TR: 0.98487 | ACC TE: 0.84616 | SIGT: 11.63906 | SIGS: 15.42720 | BS: 28.15803 | BB: 4.11545 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93517 | PREC TE: 0.89541 | ACC TR: 0.91552 | ACC TE: 0.86061 | SIGT: 12.31709 | SIGS: 18.27001 | BS: 29.28912 | BB: 3.57931 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92474 | PREC TE: 0.89640 | ACC TR: 0.90070 | ACC TE: 0.86239 | SIGT: 12.74717 | SIGS: 20.32729 | BS: 25.16423 | BB: 1.73295 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91928 | PREC TE: 0.89912 | ACC TR: 0.89411 | ACC TE: 0.86498 | SIGT: 12.74958 | SIGS: 18.96763 | BS: 27.79142 | BB: 2.49589 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91875 | PREC TE: 0.89993 | ACC TR: 0.89191 | ACC TE: 0.86626 | SIGT: 12.69240 | SIGS: 20.24703 | BS: 30.78306 | BB: 3.66760 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91667 | PREC TE: 0.89937 | ACC TR: 0.88958 | ACC TE: 0.86603 | SIGT: 12.63261 | SIGS: 20.16353 | BS: 27.74932 | BB: 2.60422 | PARAMS: 0',
]
seed_12_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98635 | PREC TE: 0.88475 | ACC TR: 0.98394 | ACC TE: 0.84909 | SIGT: 11.88613 | SIGS: 16.57780 | BS: 24.50562 | BB: 2.28501 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93685 | PREC TE: 0.89841 | ACC TR: 0.91817 | ACC TE: 0.86318 | SIGT: 12.46804 | SIGS: 17.88193 | BS: 30.49362 | BB: 3.87513 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92540 | PREC TE: 0.89916 | ACC TR: 0.90176 | ACC TE: 0.86396 | SIGT: 12.48957 | SIGS: 18.86623 | BS: 28.45938 | BB: 3.02024 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92164 | PREC TE: 0.90008 | ACC TR: 0.89434 | ACC TE: 0.86498 | SIGT: 12.42085 | SIGS: 18.98030 | BS: 26.96578 | BB: 2.56474 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91803 | PREC TE: 0.89952 | ACC TR: 0.89123 | ACC TE: 0.86607 | SIGT: 12.69200 | SIGS: 18.49192 | BS: 27.90637 | BB: 2.59372 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91698 | PREC TE: 0.90045 | ACC TR: 0.88940 | ACC TE: 0.86640 | SIGT: 12.69017 | SIGS: 18.82039 | BS: 30.95750 | BB: 3.74349 | PARAMS: 0',
]
seed_93_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98782 | PREC TE: 0.88469 | ACC TR: 0.98646 | ACC TE: 0.84712 | SIGT: 11.85310 | SIGS: 15.15009 | BS: 32.14376 | BB: 5.88130 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93591 | PREC TE: 0.89444 | ACC TR: 0.91679 | ACC TE: 0.85940 | SIGT: 12.30073 | SIGS: 16.77852 | BS: 29.85547 | BB: 3.84882 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92613 | PREC TE: 0.89638 | ACC TR: 0.90319 | ACC TE: 0.86127 | SIGT: 12.27088 | SIGS: 19.58568 | BS: 32.51394 | BB: 5.20076 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92117 | PREC TE: 0.89825 | ACC TR: 0.89569 | ACC TE: 0.86252 | SIGT: 12.38119 | SIGS: 18.36808 | BS: 27.05510 | BB: 2.63839 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91847 | PREC TE: 0.89847 | ACC TR: 0.89151 | ACC TE: 0.86439 | SIGT: 12.44856 | SIGS: 18.40584 | BS: 27.32571 | BB: 2.65862 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91724 | PREC TE: 0.89802 | ACC TR: 0.88974 | ACC TE: 0.86393 | SIGT: 12.55653 | SIGS: 18.11157 | BS: 30.20671 | BB: 3.62236 | PARAMS: 0',
]
seed_45_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.99019 | PREC TE: 0.88561 | ACC TR: 0.99045 | ACC TE: 0.84784 | SIGT: 11.98193 | SIGS: 17.48087 | BS: 27.45897 | BB: 3.27914 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93487 | PREC TE: 0.89692 | ACC TR: 0.91440 | ACC TE: 0.85930 | SIGT: 12.52841 | SIGS: 18.87319 | BS: 25.48786 | BB: 1.99696 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92499 | PREC TE: 0.89804 | ACC TR: 0.90051 | ACC TE: 0.86203 | SIGT: 12.34874 | SIGS: 18.83266 | BS: 29.36374 | BB: 3.56562 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92022 | PREC TE: 0.89834 | ACC TR: 0.89381 | ACC TE: 0.86318 | SIGT: 12.65317 | SIGS: 19.92377 | BS: 26.31042 | BB: 2.12840 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91820 | PREC TE: 0.89929 | ACC TR: 0.89102 | ACC TE: 0.86465 | SIGT: 12.66954 | SIGS: 20.29696 | BS: 30.25500 | BB: 3.48421 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91642 | PREC TE: 0.89910 | ACC TR: 0.88916 | ACC TE: 0.86459 | SIGT: 12.63462 | SIGS: 18.92631 | BS: 29.74987 | BB: 3.33060 | PARAMS: 0',
]
seed_7_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98656 | PREC TE: 0.88395 | ACC TR: 0.98580 | ACC TE: 0.85024 | SIGT: 12.06432 | SIGS: 16.95087 | BS: 26.19236 | BB: 2.68452 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93631 | PREC TE: 0.89591 | ACC TR: 0.91675 | ACC TE: 0.86163 | SIGT: 12.56987 | SIGS: 18.07955 | BS: 29.16571 | BB: 3.18786 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92505 | PREC TE: 0.89763 | ACC TR: 0.90228 | ACC TE: 0.86364 | SIGT: 12.69531 | SIGS: 18.67976 | BS: 31.24832 | BB: 3.85895 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91947 | PREC TE: 0.89821 | ACC TR: 0.89319 | ACC TE: 0.86505 | SIGT: 12.92848 | SIGS: 18.95985 | BS: 31.15645 | BB: 3.48993 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91706 | PREC TE: 0.89843 | ACC TR: 0.88998 | ACC TE: 0.86613 | SIGT: 12.91331 | SIGS: 20.31803 | BS: 26.07766 | BB: 1.84471 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91692 | PREC TE: 0.89937 | ACC TR: 0.88943 | ACC TE: 0.86679 | SIGT: 12.79142 | SIGS: 19.78723 | BS: 29.69129 | BB: 3.10831 | PARAMS: 0',
]
seed_68_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98689 | PREC TE: 0.88551 | ACC TR: 0.98381 | ACC TE: 0.84938 | SIGT: 11.97002 | SIGS: 16.72038 | BS: 28.13804 | BB: 3.58570 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93486 | PREC TE: 0.89588 | ACC TR: 0.91611 | ACC TE: 0.86117 | SIGT: 12.58271 | SIGS: 17.95958 | BS: 29.39073 | BB: 3.25831 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92582 | PREC TE: 0.89798 | ACC TR: 0.90263 | ACC TE: 0.86285 | SIGT: 12.52870 | SIGS: 18.79870 | BS: 30.45265 | BB: 3.76651 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92022 | PREC TE: 0.89831 | ACC TR: 0.89503 | ACC TE: 0.86449 | SIGT: 12.58768 | SIGS: 19.52267 | BS: 31.12987 | BB: 3.97101 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91699 | PREC TE: 0.89832 | ACC TR: 0.89008 | ACC TE: 0.86406 | SIGT: 12.75982 | SIGS: 20.15398 | BS: 31.80781 | BB: 4.00208 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91621 | PREC TE: 0.90031 | ACC TR: 0.88832 | ACC TE: 0.86617 | SIGT: 12.65837 | SIGS: 19.82458 | BS: 30.95907 | BB: 3.79073 | PARAMS: 0',
]
seed_27_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98692 | PREC TE: 0.88497 | ACC TR: 0.98540 | ACC TE: 0.84630 | SIGT: 11.74632 | SIGS: 14.97705 | BS: 29.76436 | BB: 4.75000 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93561 | PREC TE: 0.89567 | ACC TR: 0.91539 | ACC TE: 0.86091 | SIGT: 12.13187 | SIGS: 18.49097 | BS: 26.01818 | BB: 2.54776 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92700 | PREC TE: 0.89686 | ACC TR: 0.90347 | ACC TE: 0.86265 | SIGT: 12.22705 | SIGS: 17.81069 | BS: 26.36837 | BB: 2.56709 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92144 | PREC TE: 0.89886 | ACC TR: 0.89513 | ACC TE: 0.86331 | SIGT: 12.24510 | SIGS: 18.45161 | BS: 30.02389 | BB: 4.01123 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91922 | PREC TE: 0.89896 | ACC TR: 0.89204 | ACC TE: 0.86561 | SIGT: 12.63319 | SIGS: 18.39182 | BS: 28.77425 | BB: 2.96316 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91704 | PREC TE: 0.89944 | ACC TR: 0.88892 | ACC TE: 0.86551 | SIGT: 12.59364 | SIGS: 19.02909 | BS: 28.39806 | BB: 2.97722 | PARAMS: 0',
]
seed_94_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98926 | PREC TE: 0.88328 | ACC TR: 0.98806 | ACC TE: 0.84915 | SIGT: 12.11745 | SIGS: 16.80825 | BS: 27.77222 | BB: 3.22611 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93476 | PREC TE: 0.89529 | ACC TR: 0.91587 | ACC TE: 0.85999 | SIGT: 12.49114 | SIGS: 18.62968 | BS: 27.09613 | BB: 2.53440 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92575 | PREC TE: 0.89832 | ACC TR: 0.90044 | ACC TE: 0.86377 | SIGT: 13.06369 | SIGS: 18.83799 | BS: 30.90947 | BB: 3.22015 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92168 | PREC TE: 0.89762 | ACC TR: 0.89584 | ACC TE: 0.86380 | SIGT: 12.88798 | SIGS: 19.26893 | BS: 27.79255 | BB: 2.35855 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91807 | PREC TE: 0.89817 | ACC TR: 0.89075 | ACC TE: 0.86373 | SIGT: 13.12690 | SIGS: 19.76616 | BS: 28.31703 | BB: 2.29065 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91590 | PREC TE: 0.89853 | ACC TR: 0.88865 | ACC TE: 0.86505 | SIGT: 12.87364 | SIGS: 19.82927 | BS: 26.98032 | BB: 2.12836 | PARAMS: 0',
]
seed_71_params_0 = [
'DEEP: False | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.98818 | PREC TE: 0.88169 | ACC TR: 0.98753 | ACC TE: 0.84715 | SIGT: 12.00756 | SIGS: 15.52582 | BS: 27.78738 | BB: 3.38134 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.93475 | PREC TE: 0.89385 | ACC TR: 0.91585 | ACC TE: 0.85943 | SIGT: 12.47204 | SIGS: 17.22635 | BS: 31.39280 | BB: 4.27462 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92600 | PREC TE: 0.89686 | ACC TR: 0.90288 | ACC TE: 0.86291 | SIGT: 12.41319 | SIGS: 18.28835 | BS: 28.77936 | BB: 3.23970 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.92118 | PREC TE: 0.89742 | ACC TR: 0.89596 | ACC TE: 0.86367 | SIGT: 12.58133 | SIGS: 19.19823 | BS: 29.88858 | BB: 3.45736 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91768 | PREC TE: 0.89900 | ACC TR: 0.89084 | ACC TE: 0.86564 | SIGT: 12.81541 | SIGS: 18.42227 | BS: 31.28525 | BB: 3.69886 | PARAMS: 0',
'DEEP: False | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91627 | PREC TE: 0.89848 | ACC TR: 0.88925 | ACC TE: 0.86597 | SIGT: 12.76879 | SIGS: 20.12619 | BS: 28.61731 | BB: 2.75141 | PARAMS: 0',
]


def process_data(data):
    all_lists = []
    
    deep_values = []
    frs_values = []
    frg_values = []
    prec_tr_values = []
    prec_te_values = []
    acc_tr_values = []
    acc_te_values = []
    sigt_values = []
    sigs_values = []
    
    
    for seed in data:


        for item in seed:
            columns = item.split(" | ")
            #deep_values.append(float(columns[0].split(": ")[1]))
            # frs_values.append(float(columns[1].split(": ")[1]))
            # frg_values.append(float(columns[2].split(": ")[1]))
            prec_tr_values.append(float(columns[3].split(": ")[1]))
            prec_te_values.append(float(columns[4].split(": ")[1]))
            acc_tr_values.append(float(columns[5].split(": ")[1]))
            acc_te_values.append(float(columns[6].split(": ")[1]))
            sigt_values.append(float(columns[7].split(": ")[1]))
            sigs_values.append(float(columns[8].split(": ")[1]))
        # print(acc_te_values)
        # exit()
        # print("FRA:", fra_values)
        # print("ACC TR:", acc_tr_values)
        # print("ACC TE:", acc_te_values)
        # print("SIG:", sig_values)
    all_lists = [prec_tr_values, prec_te_values, acc_tr_values, acc_te_values, sigt_values, sigs_values]
    
    means_all = []
    stds_all = []
    
    for lst in all_lists:
        means = []
        stds = []
        for i in range(6):
            #print(lst[i::10])
            every_10th = lst[i::6] 
            mean = np.mean(every_10th)
            std = np.std(every_10th)
            means.append(mean)
            stds.append(std)
            # print(mean)
            # print(std)
        # for i in range(10):
        #     print(f"Mean of every 10th value starting from index {i}: {means[i]}")
        #     print(f"Variance of every 10th value starting from index {i}: {variances[i]}")
        #     print()     
        means_all.append(means)
        stds_all.append(stds)
    return means_all, stds_all
PATH_SAVE = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/mlp_output/classification_result/'

LoadStyle()
letexTitle = ROOT.TLatex()
letexTitle.SetTextSize(0.05)
letexTitle.SetNDC()
letexTitle.SetTextFont(42)


seeds_0 = [seed_32_params_0, seed_56_params_0, seed_17_params_0, seed_98_params_0, seed_42_params_0, seed_73_params_0, seed_5_params_0, seed_61_params_0, seed_89_params_0, seed_25_params_0, seed_10_params_0, seed_3_params_0, seed_12_params_0, seed_93_params_0, seed_45_params_0, seed_7_params_0,  seed_68_params_0, seed_27_params_0, seed_94_params_0, seed_71_params_0]
# seeds_1 = [seed_32_params_1, seed_56_params_1, seed_17_params_1, seed_98_params_1, seed_42_params_1, seed_73_params_1, seed_5_params_1, seed_61_params_1, seed_89_params_1, seed_25_params_1, seed_10_params_1, seed_3_params_1, seed_12_params_1, seed_93_params_1, seed_45_params_1, seed_7_params_1,  seed_68_params_1, seed_27_params_1, seed_94_params_1, seed_71_params_1]
# seeds_962 = [seed_32_params_962, seed_56_params_962, seed_17_params_962, seed_98_params_962, seed_42_params_962, seed_73_params_962, seed_5_params_962, seed_61_params_962, seed_89_params_962, seed_25_params_962, seed_10_params_962, seed_3_params_962, seed_12_params_962, seed_93_params_962, seed_45_params_962, seed_7_params_962,  seed_68_params_962, seed_27_params_962, seed_94_params_962, seed_71_params_962]
# seeds_70146 = [seed_32_params_70146, seed_56_params_70146, seed_17_params_70146, seed_98_params_70146, seed_42_params_70146, seed_73_params_70146, seed_5_params_70146, seed_61_params_70146, seed_89_params_70146, seed_25_params_70146, seed_10_params_70146, seed_3_params_70146, seed_12_params_70146, seed_93_params_70146, seed_45_params_70146, seed_7_params_70146,  seed_68_params_70146, seed_27_params_70146, seed_94_params_70146, seed_71_params_70146]


means_all_0, vars_all_0 = process_data(seeds_0)
# means_all_1, vars_all_1 = process_data(seeds_1)
# means_all_962, vars_all_962 = process_data(seeds_962)
# means_all_70146, vars_all_70146 = process_data(seeds_70146)


options = {0: 'Train_Precision', 1: 'Test_Precision',2: 'Train_Accuaracy', 3: 'Test_Accuracy', 4: 'Significance_True', 5: 'Significance_Simple'}
labels = {0: 'Train precision', 1: 'Test precision',2: 'Train accuaracy', 3: 'Test accuracy', 4: 'Significance log', 5: 'Significance simple'}
fractions = [1, 2, 3, 4, 5, 6]

canvas1 = ROOT.TCanvas("canvas1", "Data Plot")
#canvas2 = ROOT.TCanvas("canvas2", "Data Plot")

for i in range(6):
    
    # min_y = min(min(means_all_962[i]), min(means_all_70146[i]))- vars_all_962[i][0]   
    # max_y = max(max(means_all_962[i]), max(means_all_70146[i]))+ vars_all_962[i][-1]  
    min_y = min(min(means_all_0[i]), min(means_all_0[i]))- vars_all_0[i][0]   
    max_y = max(max(means_all_0[i]), max(means_all_0[i]))+ vars_all_0[i][-1]  
    
    x_min_sim = 0
    x_max_sim = 1
    x_min_gen = 1
    x_max_gen = 6
    # y_min = min(min(means_all_962[i]), min(means_all_70146[i]))- vars_all_962[i][0] 
    # y_max = max(max(means_all_962[i]), max(means_all_70146[i]))+ vars_all_962[i][-1]
    y_min = min(min(means_all_0[i]), min(means_all_0[i]))- vars_all_0[i][0] 
    y_max = max(max(means_all_0[i]), max(means_all_0[i]))+ vars_all_0[i][-1]


    rectangle_sim = ROOT.TPolyLine(4)
    rectangle_sim.SetPoint(0, x_min_sim, y_min)
    rectangle_sim.SetPoint(1, x_max_sim, y_min)
    rectangle_sim.SetPoint(2, x_max_sim, y_max)
    rectangle_sim.SetPoint(3, x_min_sim, y_max)
    #rectangle_sim.SetPoint(4, x_min_sim, y_min)
    rectangle_gen = ROOT.TPolyLine(4)
    rectangle_gen.SetPoint(0, x_min_gen, y_min)
    rectangle_gen.SetPoint(1, x_max_gen, y_min)
    rectangle_gen.SetPoint(2, x_max_gen, y_max)
    rectangle_gen.SetPoint(3, x_min_gen, y_max)
    
    rectangle_sim.SetFillColorAlpha(ROOT.kGreen, 0.1) 
    rectangle_gen.SetFillColorAlpha(ROOT.kViolet, 0.1)
    
    vars_all_0[i] = np.array(vars_all_0[i])
    # vars_all_1[i] = np.array(vars_all_1[i])/2
    # vars_all_962[i] = np.array(vars_all_962[i])/2
    # vars_all_70146[i] = np.array(vars_all_70146[i])/2
    
    graph_0 = ROOT.TGraphErrors(len(fractions), array.array('d', fractions), array.array('d', means_all_0[i]), array.array('d', [0]*len(fractions)), array.array('d', vars_all_0[i]))
    # graph_1 = ROOT.TGraphErrors(len(fractions), array.array('d', fractions), array.array('d', means_all_1[i]), array.array('d', [0]*len(fractions)), array.array('d', vars_all_1[i]))
    # graph_962 = ROOT.TGraphErrors(len(fractions), array.array('d', fractions), array.array('d', means_all_962[i]), array.array('d', [0]*len(fractions)), array.array('d', vars_all_962[i]))
    # graph_70146 = ROOT.TGraphErrors(len(fractions), array.array('d', fractions), array.array('d', means_all_70146[i]), array.array('d', [0]*len(fractions)), array.array('d', vars_all_70146[i]))
    
    print(vars_all_0[i])
    
    y_axis = graph_0.GetYaxis()
    x_axis = graph_0.GetXaxis()
    # y_axis = graph_962.GetYaxis()
    # x_axis = graph_962.GetXaxis()

    y_axis.SetRangeUser(min_y, max_y)
    x_axis.SetRangeUser(0, 6)
    
    
    graph_0.SetLineWidth(2)
    # graph_1.SetLineWidth(2)
    # graph_962.SetLineWidth(3)
    # graph_70146.SetLineWidth(3)

    graph_0.SetMarkerStyle(20)
    graph_0.SetMarkerSize(1)
    # graph_1.SetMarkerStyle(20)
    # graph_1.SetMarkerSize(1)
    # graph_962.SetMarkerStyle(20)
    # graph_962.SetMarkerSize(1)
    # graph_70146.SetMarkerStyle(20)
    # graph_70146.SetMarkerSize(1)

    graph_0.SetLineColor(ROOT.kBlue)
    # graph_1.SetLineColor(ROOT.kRed)
    # graph_962.SetLineColor(ROOT.kBlue)
    # graph_70146.SetLineColor(ROOT.kRed)

    graph_0.Draw("APL")
    # graph_1.Draw("PL SAME")

    # graph_962.Draw("APL")
    # graph_70146.Draw("PL SAME")
    rectangle_sim.Draw("F SAME")
    rectangle_gen.Draw("F SAME")

    graph_0.GetXaxis().SetTitle("Fractions")
    graph_0.GetYaxis().SetTitle(f"{labels[i]}")
    canvas1.SetTitle("")
    graph_0.SetTitle("")
    # graph_1.SetTitle("")
    
    # graph_962.GetXaxis().SetTitle("Fractions")
    # graph_962.GetYaxis().SetTitle(f"{labels[i]}")
    # canvas1.SetTitle("")
    # graph_962.SetTitle("")
    # graph_70146.SetTitle("")

    legend1 = ROOT.TLegend(0.7, 0.2, 0.9, 0.4)
    legend1.AddEntry(graph_0, "XGB 0", "lep")
    # legend1.AddEntry(graph_1, "XGB 1", "lep")
    legend1.SetBorderSize(0)
    legend1.Draw()
    
    # legend2 = ROOT.TLegend(0.7, 0.2, 0.9, 0.4)
    # legend2.AddEntry(graph_962, "MLP 962", "lep")
    # legend2.AddEntry(graph_70146, "MLP 70146", "lep")
    # legend2.SetBorderSize(0)
    # legend2.Draw()

    latex = ROOT.TLatex()
    latex.SetTextSize(0.052) 
    latex.SetTextFont(52)
    latex.DrawLatexNDC(0.35, 0.29, "Filter efficiency: 40.3%")
    latex.DrawLatexNDC(0.35, 0.23, "Cross Section: 0.1 pb")

    canvas1.SetGrid()
    canvas1.Update()
    canvas1.Draw()
    canvas1.Print(f'{PATH_SAVE}{options[i]}_root_ext_xgb_01.pdf')
    
    # canvas2.SetGrid()
    # canvas2.Update()
    # canvas2.Draw()
    # canvas2.Print(f'{PATH_SAVE}{options[i]}_root_ext_mlp.png')
