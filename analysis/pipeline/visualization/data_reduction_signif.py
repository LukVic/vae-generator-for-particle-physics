import matplotlib.pyplot as plt
import numpy as np


seed_03_params_962 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.88793 | PREC TE: 0.88121 | ACC TR: 0.87087 | ACC TE: 0.86368 | SIGT: 3.66978 | SIGS: 2.92901 | PARAMS: 962',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.89164 | PREC TE: 0.88639 | ACC TR: 0.87209 | ACC TE: 0.87014 | SIGT: 3.73807 | SIGS: 3.28501 | PARAMS: 962',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.89856 | PREC TE: 0.89351 | ACC TR: 0.87701 | ACC TE: 0.87434 | SIGT: 3.81528 | SIGS: 3.39903 | PARAMS: 962',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.89950 | PREC TE: 0.89404 | ACC TR: 0.87721 | ACC TE: 0.87644 | SIGT: 3.84515 | SIGS: 3.52528 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.90055 | PREC TE: 0.89550 | ACC TR: 0.87742 | ACC TE: 0.87599 | SIGT: 3.88642 | SIGS: 3.57435 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.89506 | PREC TE: 0.91862 | ACC TR: 0.86621 | ACC TE: 0.88035 | SIGT: 3.95762 | SIGS: 3.72155 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.89784 | PREC TE: 0.92502 | ACC TR: 0.86598 | ACC TE: 0.88065 | SIGT: 3.99108 | SIGS: 3.67689 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.89599 | PREC TE: 0.92503 | ACC TR: 0.86595 | ACC TE: 0.88080 | SIGT: 3.93297 | SIGS: 3.70293 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.89696 | PREC TE: 0.92631 | ACC TR: 0.86582 | ACC TE: 0.87915 | SIGT: 3.97090 | SIGS: 3.75263 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.89667 | PREC TE: 0.92748 | ACC TR: 0.86613 | ACC TE: 0.87975 | SIGT: 3.97317 | SIGS: 3.75337 | PARAMS: 962',    
]
seed_71_params_962 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.88549 | PREC TE: 0.88163 | ACC TR: 0.86937 | ACC TE: 0.86879 | SIGT: 3.81630 | SIGS: 3.18627 | PARAMS: 962',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.89315 | PREC TE: 0.89127 | ACC TR: 0.87331 | ACC TE: 0.87374 | SIGT: 3.89690 | SIGS: 3.47256 | PARAMS: 962',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.89684 | PREC TE: 0.89747 | ACC TR: 0.87513 | ACC TE: 0.87629 | SIGT: 3.90943 | SIGS: 3.40944 | PARAMS: 962',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.89757 | PREC TE: 0.89858 | ACC TR: 0.87604 | ACC TE: 0.87750 | SIGT: 3.93575 | SIGS: 3.45074 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.89975 | PREC TE: 0.90023 | ACC TR: 0.87769 | ACC TE: 0.87720 | SIGT: 3.94371 | SIGS: 3.62784 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.89650 | PREC TE: 0.91967 | ACC TR: 0.86929 | ACC TE: 0.87855 | SIGT: 3.97441 | SIGS: 3.69435 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.89557 | PREC TE: 0.92266 | ACC TR: 0.86762 | ACC TE: 0.87735 | SIGT: 3.99676 | SIGS: 3.89130 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.89663 | PREC TE: 0.92681 | ACC TR: 0.86693 | ACC TE: 0.87945 | SIGT: 4.00483 | SIGS: 3.94886 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.89781 | PREC TE: 0.92723 | ACC TR: 0.86625 | ACC TE: 0.87705 | SIGT: 4.01242 | SIGS: 3.94606 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.89727 | PREC TE: 0.92709 | ACC TR: 0.86594 | ACC TE: 0.87720 | SIGT: 4.02520 | SIGS: 3.99900 | PARAMS: 962',
]
seed_94_params_962 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.88006 | PREC TE: 0.89067 | ACC TR: 0.86674 | ACC TE: 0.86789 | SIGT: 3.74297 | SIGS: 3.11618 | PARAMS: 962',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.89141 | PREC TE: 0.89360 | ACC TR: 0.87603 | ACC TE: 0.87059 | SIGT: 3.79844 | SIGS: 3.15589 | PARAMS: 962',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.89703 | PREC TE: 0.89688 | ACC TR: 0.87720 | ACC TE: 0.87254 | SIGT: 3.85593 | SIGS: 3.20063 | PARAMS: 962',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.90019 | PREC TE: 0.89901 | ACC TR: 0.87871 | ACC TE: 0.87284 | SIGT: 3.88082 | SIGS: 3.31095 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.90058 | PREC TE: 0.90171 | ACC TR: 0.87826 | ACC TE: 0.87539 | SIGT: 3.90304 | SIGS: 3.37583 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.89641 | PREC TE: 0.92179 | ACC TR: 0.86802 | ACC TE: 0.87960 | SIGT: 3.98754 | SIGS: 3.48456 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.89620 | PREC TE: 0.92712 | ACC TR: 0.86724 | ACC TE: 0.88110 | SIGT: 4.01969 | SIGS: 3.57069 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.89746 | PREC TE: 0.93013 | ACC TR: 0.86690 | ACC TE: 0.88065 | SIGT: 4.05337 | SIGS: 3.71512 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.89785 | PREC TE: 0.93219 | ACC TR: 0.86631 | ACC TE: 0.88200 | SIGT: 4.05805 | SIGS: 3.72508 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.89828 | PREC TE: 0.93244 | ACC TR: 0.86667 | ACC TE: 0.88110 | SIGT: 4.06560 | SIGS: 3.78625 | PARAMS: 962',
]
seed_27_params_962 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.88623 | PREC TE: 0.88481 | ACC TR: 0.86862 | ACC TE: 0.87014 | SIGT: 3.74799 | SIGS: 3.12838 | PARAMS: 962',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.89264 | PREC TE: 0.89523 | ACC TR: 0.87040 | ACC TE: 0.87494 | SIGT: 3.79509 | SIGS: 3.13084 | PARAMS: 962',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.89754 | PREC TE: 0.89796 | ACC TR: 0.87570 | ACC TE: 0.87810 | SIGT: 3.87206 | SIGS: 3.17990 | PARAMS: 962',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.89949 | PREC TE: 0.89895 | ACC TR: 0.87679 | ACC TE: 0.87825 | SIGT: 3.89377 | SIGS: 3.17368 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.90024 | PREC TE: 0.89818 | ACC TR: 0.87780 | ACC TE: 0.87765 | SIGT: 3.90964 | SIGS: 3.27764 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.89699 | PREC TE: 0.91890 | ACC TR: 0.86903 | ACC TE: 0.87855 | SIGT: 3.95738 | SIGS: 3.34458 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.89524 | PREC TE: 0.92244 | ACC TR: 0.86729 | ACC TE: 0.87990 | SIGT: 3.96982 | SIGS: 3.46141 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.89615 | PREC TE: 0.92705 | ACC TR: 0.86624 | ACC TE: 0.88035 | SIGT: 3.97775 | SIGS: 3.57649 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.89658 | PREC TE: 0.92897 | ACC TR: 0.86590 | ACC TE: 0.88035 | SIGT: 3.97637 | SIGS: 3.59188 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.89787 | PREC TE: 0.93005 | ACC TR: 0.86610 | ACC TE: 0.87795 | SIGT: 3.98664 | SIGS: 3.57095 | PARAMS: 962',
]
seed_68_params_962 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.88133 | PREC TE: 0.87599 | ACC TR: 0.86730 | ACC TE: 0.86308 | SIGT: 3.72442 | SIGS: 3.03614 | PARAMS: 962',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.89345 | PREC TE: 0.89489 | ACC TR: 0.86984 | ACC TE: 0.87104 | SIGT: 3.79092 | SIGS: 3.14324 | PARAMS: 962',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.89381 | PREC TE: 0.89411 | ACC TR: 0.87363 | ACC TE: 0.87359 | SIGT: 3.83964 | SIGS: 3.27913 | PARAMS: 962',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.89869 | PREC TE: 0.89786 | ACC TR: 0.87712 | ACC TE: 0.87494 | SIGT: 3.90328 | SIGS: 3.36162 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.90101 | PREC TE: 0.89989 | ACC TR: 0.87848 | ACC TE: 0.87675 | SIGT: 3.94322 | SIGS: 3.41642 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.89555 | PREC TE: 0.91996 | ACC TR: 0.86678 | ACC TE: 0.87990 | SIGT: 4.01075 | SIGS: 3.50909 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.89848 | PREC TE: 0.92802 | ACC TR: 0.86675 | ACC TE: 0.88050 | SIGT: 4.05360 | SIGS: 3.56019 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.89684 | PREC TE: 0.92937 | ACC TR: 0.86716 | ACC TE: 0.88125 | SIGT: 4.06825 | SIGS: 3.60857 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.89877 | PREC TE: 0.93206 | ACC TR: 0.86697 | ACC TE: 0.88050 | SIGT: 4.07056 | SIGS: 3.60892 | PARAMS: 962',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.89779 | PREC TE: 0.93223 | ACC TR: 0.86627 | ACC TE: 0.88050 | SIGT: 4.07337 | SIGS: 3.62993 | PARAMS: 962',
]


seed_03_params_70146 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91510 | PREC TE: 0.90820 | ACC TR: 0.89208 | ACC TE: 0.87930 | SIGT: 3.94143 | SIGS: 3.54086 | PARAMS: 70146',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91389 | PREC TE: 0.91027 | ACC TR: 0.88673 | ACC TE: 0.88500 | SIGT: 4.06722 | SIGS: 3.79676 | PARAMS: 70146',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91739 | PREC TE: 0.90987 | ACC TR: 0.89071 | ACC TE: 0.88530 | SIGT: 4.10724 | SIGS: 3.77706 | PARAMS: 70146',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91716 | PREC TE: 0.90891 | ACC TR: 0.89030 | ACC TE: 0.88455 | SIGT: 4.11765 | SIGS: 3.88018 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91663 | PREC TE: 0.90987 | ACC TR: 0.89001 | ACC TE: 0.88395 | SIGT: 4.10382 | SIGS: 3.88018 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90498 | PREC TE: 0.92361 | ACC TR: 0.87719 | ACC TE: 0.88575 | SIGT: 4.11698 | SIGS: 4.13814 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90861 | PREC TE: 0.92963 | ACC TR: 0.87551 | ACC TE: 0.88230 | SIGT: 4.09837 | SIGS: 4.13602 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90819 | PREC TE: 0.93061 | ACC TR: 0.87547 | ACC TE: 0.88245 | SIGT: 4.13599 | SIGS: 4.17598 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90953 | PREC TE: 0.93265 | ACC TR: 0.87522 | ACC TE: 0.88170 | SIGT: 4.14279 | SIGS: 4.22268 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90841 | PREC TE: 0.93296 | ACC TR: 0.87589 | ACC TE: 0.88335 | SIGT: 4.15071 | SIGS: 4.09794 | PARAMS: 70146',
]
seed_71_params_70146 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.92216 | PREC TE: 0.90746 | ACC TR: 0.89640 | ACC TE: 0.88185 | SIGT: 4.08348 | SIGS: 3.89492 | PARAMS: 70146',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91881 | PREC TE: 0.90873 | ACC TR: 0.89330 | ACC TE: 0.88155 | SIGT: 4.14314 | SIGS: 4.07340 | PARAMS: 70146',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91805 | PREC TE: 0.91234 | ACC TR: 0.89046 | ACC TE: 0.88515 | SIGT: 4.15278 | SIGS: 4.02979 | PARAMS: 70146',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91624 | PREC TE: 0.91217 | ACC TR: 0.89007 | ACC TE: 0.88635 | SIGT: 4.14544 | SIGS: 4.09102 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91651 | PREC TE: 0.91257 | ACC TR: 0.88997 | ACC TE: 0.88590 | SIGT: 4.16761 | SIGS: 4.19499 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90743 | PREC TE: 0.92353 | ACC TR: 0.88002 | ACC TE: 0.88320 | SIGT: 4.16693 | SIGS: 4.28200 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90723 | PREC TE: 0.92797 | ACC TR: 0.87723 | ACC TE: 0.88170 | SIGT: 4.16019 | SIGS: 4.35158 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91002 | PREC TE: 0.93280 | ACC TR: 0.87658 | ACC TE: 0.88155 | SIGT: 4.18882 | SIGS: 4.34918 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91012 | PREC TE: 0.93427 | ACC TR: 0.87550 | ACC TE: 0.88140 | SIGT: 4.19244 | SIGS: 4.24375 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90980 | PREC TE: 0.93426 | ACC TR: 0.87536 | ACC TE: 0.88125 | SIGT: 4.21880 | SIGS: 4.32907 | PARAMS: 70146',
]
seed_94_params_70146 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91644 | PREC TE: 0.91582 | ACC TR: 0.89452 | ACC TE: 0.87975 | SIGT: 3.99519 | SIGS: 3.62976 | PARAMS: 70146',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91315 | PREC TE: 0.91479 | ACC TR: 0.89142 | ACC TE: 0.88440 | SIGT: 4.06457 | SIGS: 3.72709 | PARAMS: 70146',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91488 | PREC TE: 0.91581 | ACC TR: 0.89033 | ACC TE: 0.88560 | SIGT: 4.09532 | SIGS: 3.73974 | PARAMS: 70146',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91672 | PREC TE: 0.91634 | ACC TR: 0.89044 | ACC TE: 0.88785 | SIGT: 4.10093 | SIGS: 3.83581 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91593 | PREC TE: 0.91599 | ACC TR: 0.88963 | ACC TE: 0.88740 | SIGT: 4.08458 | SIGS: 3.81395 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90936 | PREC TE: 0.93149 | ACC TR: 0.87727 | ACC TE: 0.88710 | SIGT: 4.11863 | SIGS: 3.95875 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90900 | PREC TE: 0.93442 | ACC TR: 0.87596 | ACC TE: 0.88725 | SIGT: 4.15617 | SIGS: 3.97968 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90610 | PREC TE: 0.93311 | ACC TR: 0.87619 | ACC TE: 0.88710 | SIGT: 4.15415 | SIGS: 4.10150 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90762 | PREC TE: 0.93548 | ACC TR: 0.87537 | ACC TE: 0.88620 | SIGT: 4.14654 | SIGS: 4.00823 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91009 | PREC TE: 0.93763 | ACC TR: 0.87602 | ACC TE: 0.88620 | SIGT: 4.14881 | SIGS: 3.91338 | PARAMS: 70146',
]
seed_27_params_70146 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91601 | PREC TE: 0.91006 | ACC TR: 0.89058 | ACC TE: 0.88440 | SIGT: 4.07822 | SIGS: 3.74961 | PARAMS: 70146',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91586 | PREC TE: 0.91370 | ACC TR: 0.89020 | ACC TE: 0.88680 | SIGT: 4.06415 | SIGS: 3.64427 | PARAMS: 70146',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91581 | PREC TE: 0.91129 | ACC TR: 0.89090 | ACC TE: 0.88770 | SIGT: 4.09654 | SIGS: 3.75068 | PARAMS: 70146',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91573 | PREC TE: 0.91303 | ACC TR: 0.88969 | ACC TE: 0.88891 | SIGT: 4.08490 | SIGS: 3.81587 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91579 | PREC TE: 0.91287 | ACC TR: 0.88967 | ACC TE: 0.88876 | SIGT: 4.09302 | SIGS: 3.86828 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90748 | PREC TE: 0.92580 | ACC TR: 0.87943 | ACC TE: 0.88740 | SIGT: 4.10236 | SIGS: 3.92285 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90432 | PREC TE: 0.92725 | ACC TR: 0.87631 | ACC TE: 0.88605 | SIGT: 4.09768 | SIGS: 3.84214 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90640 | PREC TE: 0.93088 | ACC TR: 0.87571 | ACC TE: 0.88560 | SIGT: 4.09890 | SIGS: 3.82855 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90573 | PREC TE: 0.93120 | ACC TR: 0.87581 | ACC TE: 0.88560 | SIGT: 4.11459 | SIGS: 4.03643 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90577 | PREC TE: 0.93110 | ACC TR: 0.87586 | ACC TE: 0.88440 | SIGT: 4.13075 | SIGS: 3.97978 | PARAMS: 70146',
]
seed_68_params_70146 = [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91756 | PREC TE: 0.90830 | ACC TR: 0.89339 | ACC TE: 0.88155 | SIGT: 4.02196 | SIGS: 3.55852 | PARAMS: 70146',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91220 | PREC TE: 0.91352 | ACC TR: 0.88682 | ACC TE: 0.88365 | SIGT: 4.05332 | SIGS: 3.64784 | PARAMS: 70146',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91370 | PREC TE: 0.91375 | ACC TR: 0.88946 | ACC TE: 0.88590 | SIGT: 4.14283 | SIGS: 3.69476 | PARAMS: 70146',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91618 | PREC TE: 0.91507 | ACC TR: 0.89030 | ACC TE: 0.88710 | SIGT: 4.15509 | SIGS: 3.81524 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91699 | PREC TE: 0.91441 | ACC TR: 0.89062 | ACC TE: 0.88650 | SIGT: 4.14940 | SIGS: 3.79851 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90899 | PREC TE: 0.93023 | ACC TR: 0.87754 | ACC TE: 0.88740 | SIGT: 4.18546 | SIGS: 3.79726 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90858 | PREC TE: 0.93300 | ACC TR: 0.87591 | ACC TE: 0.88575 | SIGT: 4.22735 | SIGS: 3.77054 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90749 | PREC TE: 0.93385 | ACC TR: 0.87596 | ACC TE: 0.88425 | SIGT: 4.20474 | SIGS: 3.83372 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90703 | PREC TE: 0.93522 | ACC TR: 0.87592 | ACC TE: 0.88500 | SIGT: 4.20869 | SIGS: 3.92351 | PARAMS: 70146',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90957 | PREC TE: 0.93820 | ACC TR: 0.87589 | ACC TE: 0.88500 | SIGT: 4.20337 | SIGS: 3.97971 | PARAMS: 70146',
]


seed_03_params_536066= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91477 | PREC TE: 0.91019 | ACC TR: 0.88889 | ACC TE: 0.87870 | SIGT: 3.87501 | SIGS: 3.34308 | PARAMS: 536066',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91310 | PREC TE: 0.90953 | ACC TR: 0.88532 | ACC TE: 0.88350 | SIGT: 4.01549 | SIGS: 3.65856 | PARAMS: 536066',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91642 | PREC TE: 0.91008 | ACC TR: 0.88908 | ACC TE: 0.88590 | SIGT: 4.04449 | SIGS: 3.69471 | PARAMS: 536066',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91544 | PREC TE: 0.90783 | ACC TR: 0.88852 | ACC TE: 0.88395 | SIGT: 4.05119 | SIGS: 3.82865 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91516 | PREC TE: 0.90919 | ACC TR: 0.88811 | ACC TE: 0.88440 | SIGT: 4.06842 | SIGS: 3.81523 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90504 | PREC TE: 0.92352 | ACC TR: 0.87466 | ACC TE: 0.88470 | SIGT: 4.11695 | SIGS: 4.02545 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90695 | PREC TE: 0.92793 | ACC TR: 0.87370 | ACC TE: 0.88305 | SIGT: 4.11727 | SIGS: 4.14873 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90446 | PREC TE: 0.92776 | ACC TR: 0.87305 | ACC TE: 0.88290 | SIGT: 4.13341 | SIGS: 3.99227 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90547 | PREC TE: 0.93047 | ACC TR: 0.87284 | ACC TE: 0.88275 | SIGT: 4.11563 | SIGS: 4.08893 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90635 | PREC TE: 0.93233 | ACC TR: 0.87375 | ACC TE: 0.88365 | SIGT: 4.13387 | SIGS: 4.10413 | PARAMS: 536066',
]
seed_71_params_536066= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.92033 | PREC TE: 0.90885 | ACC TR: 0.89114 | ACC TE: 0.88125 | SIGT: 4.01509 | SIGS: 3.99058 | PARAMS: 536066',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91844 | PREC TE: 0.90916 | ACC TR: 0.89048 | ACC TE: 0.88140 | SIGT: 4.10221 | SIGS: 4.08658 | PARAMS: 536066',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91596 | PREC TE: 0.91075 | ACC TR: 0.88827 | ACC TE: 0.88395 | SIGT: 4.09521 | SIGS: 4.05774 | PARAMS: 536066',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91474 | PREC TE: 0.90890 | ACC TR: 0.88861 | ACC TE: 0.88440 | SIGT: 4.12122 | SIGS: 4.05509 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91495 | PREC TE: 0.90969 | ACC TR: 0.88827 | ACC TE: 0.88365 | SIGT: 4.13486 | SIGS: 4.07788 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90717 | PREC TE: 0.92510 | ACC TR: 0.87744 | ACC TE: 0.88320 | SIGT: 4.13677 | SIGS: 4.22455 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90607 | PREC TE: 0.92723 | ACC TR: 0.87524 | ACC TE: 0.88050 | SIGT: 4.13905 | SIGS: 4.17869 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90539 | PREC TE: 0.92900 | ACC TR: 0.87460 | ACC TE: 0.88065 | SIGT: 4.15154 | SIGS: 4.12987 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90706 | PREC TE: 0.93176 | ACC TR: 0.87400 | ACC TE: 0.88080 | SIGT: 4.18560 | SIGS: 4.16318 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90557 | PREC TE: 0.93085 | ACC TR: 0.87379 | ACC TE: 0.87960 | SIGT: 4.19394 | SIGS: 4.15994 | PARAMS: 536066',
]
seed_94_params_536066= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91366 | PREC TE: 0.91880 | ACC TR: 0.88758 | ACC TE: 0.88065 | SIGT: 3.97834 | SIGS: 3.63027 | PARAMS: 536066',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91120 | PREC TE: 0.91542 | ACC TR: 0.88786 | ACC TE: 0.88320 | SIGT: 4.03404 | SIGS: 3.77841 | PARAMS: 536066',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91321 | PREC TE: 0.91590 | ACC TR: 0.88852 | ACC TE: 0.88500 | SIGT: 4.04785 | SIGS: 3.76443 | PARAMS: 536066',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91471 | PREC TE: 0.91542 | ACC TR: 0.88805 | ACC TE: 0.88620 | SIGT: 4.09207 | SIGS: 3.89795 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91478 | PREC TE: 0.91610 | ACC TR: 0.88796 | ACC TE: 0.88545 | SIGT: 4.06804 | SIGS: 3.97963 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90678 | PREC TE: 0.92910 | ACC TR: 0.87598 | ACC TE: 0.88725 | SIGT: 4.11480 | SIGS: 3.98485 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90464 | PREC TE: 0.93069 | ACC TR: 0.87417 | ACC TE: 0.88710 | SIGT: 4.12765 | SIGS: 3.97604 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90374 | PREC TE: 0.93225 | ACC TR: 0.87356 | ACC TE: 0.88650 | SIGT: 4.15787 | SIGS: 4.06114 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90457 | PREC TE: 0.93416 | ACC TR: 0.87308 | ACC TE: 0.88605 | SIGT: 4.14898 | SIGS: 4.06179 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90506 | PREC TE: 0.93499 | ACC TR: 0.87355 | ACC TE: 0.88620 | SIGT: 4.15565 | SIGS: 4.14375 | PARAMS: 536066',
]
seed_27_params_536066= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91413 | PREC TE: 0.91143 | ACC TR: 0.88682 | ACC TE: 0.88350 | SIGT: 4.04700 | SIGS: 3.59066 | PARAMS: 536066',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91477 | PREC TE: 0.91372 | ACC TR: 0.88692 | ACC TE: 0.88560 | SIGT: 4.03903 | SIGS: 3.70404 | PARAMS: 536066',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91364 | PREC TE: 0.91243 | ACC TR: 0.88733 | ACC TE: 0.88740 | SIGT: 4.05409 | SIGS: 3.70299 | PARAMS: 536066',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91398 | PREC TE: 0.91224 | ACC TR: 0.88716 | ACC TE: 0.88695 | SIGT: 4.05175 | SIGS: 3.69113 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91459 | PREC TE: 0.91139 | ACC TR: 0.88815 | ACC TE: 0.88725 | SIGT: 4.07741 | SIGS: 3.79724 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90767 | PREC TE: 0.92662 | ACC TR: 0.87758 | ACC TE: 0.88605 | SIGT: 4.10425 | SIGS: 3.81914 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90346 | PREC TE: 0.92786 | ACC TR: 0.87439 | ACC TE: 0.88575 | SIGT: 4.10126 | SIGS: 3.75267 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90313 | PREC TE: 0.92966 | ACC TR: 0.87290 | ACC TE: 0.88455 | SIGT: 4.12250 | SIGS: 3.83355 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90471 | PREC TE: 0.93212 | ACC TR: 0.87348 | ACC TE: 0.88500 | SIGT: 4.11430 | SIGS: 3.81692 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90606 | PREC TE: 0.93296 | ACC TR: 0.87363 | ACC TE: 0.88335 | SIGT: 4.11025 | SIGS: 3.79712 | PARAMS: 536066',
]
seed_68_params_536066= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.91423 | PREC TE: 0.90873 | ACC TR: 0.88438 | ACC TE: 0.88020 | SIGT: 3.96545 | SIGS: 3.61249 | PARAMS: 536066',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.91020 | PREC TE: 0.91187 | ACC TR: 0.88279 | ACC TE: 0.88200 | SIGT: 4.01465 | SIGS: 3.68512 | PARAMS: 536066',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.91091 | PREC TE: 0.91190 | ACC TR: 0.88452 | ACC TE: 0.88515 | SIGT: 4.10872 | SIGS: 3.67685 | PARAMS: 536066',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.91387 | PREC TE: 0.91321 | ACC TR: 0.88767 | ACC TE: 0.88635 | SIGT: 4.13708 | SIGS: 3.72259 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.91470 | PREC TE: 0.91321 | ACC TR: 0.88811 | ACC TE: 0.88635 | SIGT: 4.13874 | SIGS: 3.72252 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.90673 | PREC TE: 0.92876 | ACC TR: 0.87553 | ACC TE: 0.88695 | SIGT: 4.19271 | SIGS: 3.82212 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.90621 | PREC TE: 0.93407 | ACC TR: 0.87422 | ACC TE: 0.88695 | SIGT: 4.21583 | SIGS: 3.74863 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.90358 | PREC TE: 0.93252 | ACC TR: 0.87404 | ACC TE: 0.88590 | SIGT: 4.22253 | SIGS: 3.79577 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90448 | PREC TE: 0.93397 | ACC TR: 0.87437 | ACC TE: 0.88575 | SIGT: 4.23080 | SIGS: 3.84640 | PARAMS: 536066',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90564 | PREC TE: 0.93506 | ACC TR: 0.87346 | ACC TE: 0.88500 | SIGT: 4.23190 | SIGS: 3.85445 | PARAMS: 536066',
]

seed_03_params_4739586= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.95310 | PREC TE: 0.91800 | ACC TR: 0.92905 | ACC TE: 0.87419 | SIGT: 3.92885 | SIGS: 3.41609 | PARAMS: 4739586',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.94085 | PREC TE: 0.91724 | ACC TR: 0.91498 | ACC TE: 0.88035 | SIGT: 4.05001 | SIGS: 3.49876 | PARAMS: 4739586',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.93578 | PREC TE: 0.91522 | ACC TR: 0.91192 | ACC TE: 0.88425 | SIGT: 4.05059 | SIGS: 3.68908 | PARAMS: 4739586',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.93248 | PREC TE: 0.91411 | ACC TR: 0.90940 | ACC TE: 0.88650 | SIGT: 4.15310 | SIGS: 3.85321 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.92851 | PREC TE: 0.91163 | ACC TR: 0.90582 | ACC TE: 0.88395 | SIGT: 4.10742 | SIGS: 4.03785 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.91152 | PREC TE: 0.92418 | ACC TR: 0.88812 | ACC TE: 0.88680 | SIGT: 4.14742 | SIGS: 3.98700 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.91611 | PREC TE: 0.93085 | ACC TR: 0.88604 | ACC TE: 0.88335 | SIGT: 4.14920 | SIGS: 4.01708 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91678 | PREC TE: 0.93314 | ACC TR: 0.88331 | ACC TE: 0.88170 | SIGT: 4.12382 | SIGS: 4.11655 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91519 | PREC TE: 0.93477 | ACC TR: 0.88208 | ACC TE: 0.88350 | SIGT: 4.18699 | SIGS: 4.16823 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91219 | PREC TE: 0.93338 | ACC TR: 0.88165 | ACC TE: 0.88455 | SIGT: 4.18920 | SIGS: 4.01065 | PARAMS: 4739586',
]
seed_71_params_4739586= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.95605 | PREC TE: 0.91951 | ACC TR: 0.93206 | ACC TE: 0.88005 | SIGT: 4.05415 | SIGS: 4.07503 | PARAMS: 4739586',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.94394 | PREC TE: 0.91624 | ACC TR: 0.91920 | ACC TE: 0.88095 | SIGT: 4.10118 | SIGS: 4.15508 | PARAMS: 4739586',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.93763 | PREC TE: 0.91683 | ACC TR: 0.91204 | ACC TE: 0.88380 | SIGT: 4.17063 | SIGS: 4.16172 | PARAMS: 4739586',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.93375 | PREC TE: 0.91749 | ACC TR: 0.90841 | ACC TE: 0.88740 | SIGT: 4.18828 | SIGS: 4.20884 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.93058 | PREC TE: 0.91628 | ACC TR: 0.90612 | ACC TE: 0.88725 | SIGT: 4.23045 | SIGS: 4.34400 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.91785 | PREC TE: 0.92762 | ACC TR: 0.89188 | ACC TE: 0.88485 | SIGT: 4.21676 | SIGS: 4.36395 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.91643 | PREC TE: 0.92992 | ACC TR: 0.88775 | ACC TE: 0.88200 | SIGT: 4.19342 | SIGS: 4.14233 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91846 | PREC TE: 0.93514 | ACC TR: 0.88503 | ACC TE: 0.88200 | SIGT: 4.22368 | SIGS: 4.17528 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91522 | PREC TE: 0.93414 | ACC TR: 0.88394 | ACC TE: 0.88185 | SIGT: 4.21443 | SIGS: 4.16961 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91412 | PREC TE: 0.93381 | ACC TR: 0.88271 | ACC TE: 0.88185 | SIGT: 4.24689 | SIGS: 4.33538 | PARAMS: 4739586',
]
seed_94_params_4739586= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.94732 | PREC TE: 0.92487 | ACC TR: 0.92774 | ACC TE: 0.87900 | SIGT: 3.97060 | SIGS: 3.58592 | PARAMS: 4739586',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.93668 | PREC TE: 0.92195 | ACC TR: 0.91657 | ACC TE: 0.88455 | SIGT: 4.05883 | SIGS: 3.84090 | PARAMS: 4739586',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.93470 | PREC TE: 0.91948 | ACC TR: 0.91248 | ACC TE: 0.88440 | SIGT: 4.05275 | SIGS: 3.85997 | PARAMS: 4739586',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.93243 | PREC TE: 0.91833 | ACC TR: 0.90883 | ACC TE: 0.88515 | SIGT: 4.06592 | SIGS: 3.92762 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.92900 | PREC TE: 0.91837 | ACC TR: 0.90677 | ACC TE: 0.88710 | SIGT: 4.06872 | SIGS: 4.19167 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.91896 | PREC TE: 0.93270 | ACC TR: 0.89026 | ACC TE: 0.88800 | SIGT: 4.14935 | SIGS: 4.11519 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.91576 | PREC TE: 0.93436 | ACC TR: 0.88826 | ACC TE: 0.88846 | SIGT: 4.20533 | SIGS: 4.17092 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91377 | PREC TE: 0.93684 | ACC TR: 0.88469 | ACC TE: 0.88876 | SIGT: 4.15690 | SIGS: 4.16607 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91580 | PREC TE: 0.94130 | ACC TR: 0.88046 | ACC TE: 0.88831 | SIGT: 4.20482 | SIGS: 4.15892 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91487 | PREC TE: 0.94021 | ACC TR: 0.88079 | ACC TE: 0.88725 | SIGT: 4.16339 | SIGS: 4.14832 | PARAMS: 4739586',
]
seed_27_params_4739586= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.95176 | PREC TE: 0.92103 | ACC TR: 0.92830 | ACC TE: 0.88140 | SIGT: 4.03833 | SIGS: 3.84452 | PARAMS: 4739586',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.93909 | PREC TE: 0.91789 | ACC TR: 0.91620 | ACC TE: 0.88380 | SIGT: 4.14034 | SIGS: 3.86784 | PARAMS: 4739586',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.93494 | PREC TE: 0.91622 | ACC TR: 0.91273 | ACC TE: 0.88515 | SIGT: 4.16050 | SIGS: 4.05592 | PARAMS: 4739586',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.92963 | PREC TE: 0.91387 | ACC TR: 0.90785 | ACC TE: 0.88560 | SIGT: 4.14560 | SIGS: 4.16185 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.92759 | PREC TE: 0.91401 | ACC TR: 0.90567 | ACC TE: 0.88695 | SIGT: 4.12566 | SIGS: 4.09101 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.91948 | PREC TE: 0.92803 | ACC TR: 0.89293 | ACC TE: 0.88770 | SIGT: 4.15388 | SIGS: 4.21056 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.91204 | PREC TE: 0.92831 | ACC TR: 0.88659 | ACC TE: 0.88725 | SIGT: 4.13638 | SIGS: 4.13752 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91370 | PREC TE: 0.93232 | ACC TR: 0.88405 | ACC TE: 0.88545 | SIGT: 4.18505 | SIGS: 4.37805 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.90816 | PREC TE: 0.92908 | ACC TR: 0.88377 | ACC TE: 0.88695 | SIGT: 4.15715 | SIGS: 4.26727 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.90815 | PREC TE: 0.92963 | ACC TR: 0.88445 | ACC TE: 0.88785 | SIGT: 4.19109 | SIGS: 4.34197 | PARAMS: 4739586',
]
seed_68_params_4739586= [
'DEEP: True | FRS: 0.2 | FRG: 0.0 | PREC TR: 0.94966 | PREC TE: 0.91954 | ACC TR: 0.92605 | ACC TE: 0.87569 | SIGT: 3.99860 | SIGS: 3.61446 | PARAMS: 4739586',
'DEEP: True | FRS: 0.4 | FRG: 0.0 | PREC TR: 0.93786 | PREC TE: 0.91938 | ACC TR: 0.91366 | ACC TE: 0.88185 | SIGT: 4.09148 | SIGS: 3.79778 | PARAMS: 4739586',
'DEEP: True | FRS: 0.6 | FRG: 0.0 | PREC TR: 0.93407 | PREC TE: 0.91819 | ACC TR: 0.91092 | ACC TE: 0.88530 | SIGT: 4.17111 | SIGS: 3.96360 | PARAMS: 4739586',
'DEEP: True | FRS: 0.8 | FRG: 0.0 | PREC TR: 0.93150 | PREC TE: 0.91739 | ACC TR: 0.90808 | ACC TE: 0.88635 | SIGT: 4.16560 | SIGS: 3.97996 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.0 | PREC TR: 0.93014 | PREC TE: 0.91570 | ACC TR: 0.90612 | ACC TE: 0.88455 | SIGT: 4.16113 | SIGS: 3.96135 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.2 | PREC TR: 0.91560 | PREC TE: 0.92853 | ACC TR: 0.88987 | ACC TE: 0.88620 | SIGT: 4.20973 | SIGS: 3.88290 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.4 | PREC TR: 0.92148 | PREC TE: 0.93938 | ACC TR: 0.88508 | ACC TE: 0.88290 | SIGT: 4.18163 | SIGS: 3.86602 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.6 | PREC TR: 0.91556 | PREC TE: 0.93762 | ACC TR: 0.88617 | ACC TE: 0.88605 | SIGT: 4.21043 | SIGS: 3.98147 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 0.8 | PREC TR: 0.91315 | PREC TE: 0.93808 | ACC TR: 0.88560 | ACC TE: 0.88770 | SIGT: 4.28512 | SIGS: 4.04454 | PARAMS: 4739586',
'DEEP: True | FRS: 1.0 | FRG: 1.0 | PREC TR: 0.91075 | PREC TE: 0.93751 | ACC TR: 0.88448 | ACC TE: 0.88891 | SIGT: 4.26467 | SIGS: 4.07244 | PARAMS: 4739586',
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

        # print("FRA:", fra_values)
        # print("ACC TR:", acc_tr_values)
        # print("ACC TE:", acc_te_values)
        # print("SIG:", sig_values)
        # Convert arrays to numpy arrays
    all_lists = [prec_tr_values, prec_te_values, acc_tr_values, acc_te_values, sigt_values, sigs_values]
    # Initialize lists to store means and variances
    prec_tr_means = []
    prec_tr_variances = []
    prec_te_values_mean = []
    prec_te_values_varianvces = []
    acc_tr_values_mean = []
    acc_tr_values_varianvces = []
    acc_te_values_mean = []
    acc_te_values_varianvces = []
    sigt_values_mean = []
    sigt_values_varianvces = []
    sigs_values_mean = []
    sigs_values_varianvces = []
    
    means_all = []
    stds_all = []
    # Extract every 10th value, then every 10th value with an incremented starting index
    
    for lst in all_lists:
        means = []
        stds = []
        for i in range(10):
            every_10th =  lst[i::10]  # Adjust the starting index
            mean = np.mean(every_10th)
            std = np.std(every_10th)
            means.append(mean)
            stds.append(std)

        # Print means and variances
        # for i in range(10):
        #     print(f"Mean of every 10th value starting from index {i}: {means[i]}")
        #     print(f"Variance of every 10th value starting from index {i}: {variances[i]}")
        #     print()     
        means_all.append(means)
        stds_all.append(stds)

    return means_all, stds_all
PATH_SAVE = '/home/lucas/Documents/KYR/msc_thesis/vae-generator-for-particle-physics/analysis/results/mlp_output/classification_result/'

# signif_frac_big_30_processed = process_data(signif_frac_big_30)
# signif_frac_big_300_processed = process_data(signif_frac_big_300)
# signif_frac_small_30_processed = process_data(signif_frac_small_30)
# signif_frac_small_300_processed = process_data(signif_frac_small_300)

# signif_frac_big_30_augmented_processed = process_data(signif_frac_big_30_augmented)
# signif_frac_small_300_augmented_processed = process_data(signif_frac_small_300_augmented)

# signif_frac_small_300_augmented_only_processed = process_data(signif_frac_small_300_augmented_only)
# signif_frac_big_30_augmented_only_processed = process_data(signif_frac_big_30_augmented_only)
# Data

seeds_962 = [seed_03_params_962,seed_27_params_962,seed_68_params_962,seed_71_params_962,seed_94_params_962]
seeds_70146 = [seed_03_params_70146,seed_27_params_70146,seed_68_params_70146,seed_71_params_70146,seed_94_params_70146]
seeds_536066 = [seed_03_params_536066,seed_27_params_536066,seed_68_params_536066,seed_71_params_536066,seed_94_params_536066]
seeds_4739586 = [seed_03_params_4739586,seed_27_params_4739586,seed_68_params_4739586,seed_71_params_4739586,seed_94_params_4739586]
means_all_962, vars_all_962 = process_data(seeds_962)
means_all_70146, vars_all_70146 = process_data(seeds_70146)
means_all_536066, vars_all_536066 = process_data(seeds_536066)
means_all_4739586, vars_all_4739586 = process_data(seeds_4739586)

options = {0: 'Train_Precision', 1: 'Test_Precision',2: 'Train_Accuaracy', 3: 'Test_Accuracy', 4: 'Significance_True', 5: 'Significance_Simple'}



for i in range(6):
    # Initialize plot
    plt.figure(figsize=(10, 6))
    
    # Loop through datasets
    # Calculate mean and standard deviation for each fraction
    fractions = np.arange(0.2, 2.2, 0.2)  # Adjust the range for the last 6 points
    print(fractions)
    # Plot mean values with error bars for each dataset
    plt.errorbar(fractions, means_all_962[i], yerr=vars_all_962[i], fmt='o-', capsize=5, markersize=8, label='Parameters 962')
    plt.errorbar(fractions, means_all_70146[i], yerr=vars_all_70146[i], fmt='o-', capsize=5, markersize=8, label='Parameters 70146')
    plt.errorbar(fractions, means_all_536066[i], yerr=vars_all_536066[i], fmt='o-', capsize=5, markersize=8, label='Parameters 536066')
    plt.errorbar(fractions, means_all_4739586[i], yerr=vars_all_4739586[i], fmt='o-', capsize=5, markersize=8, label='Parameters 4739586')

    # Add labels and title
    plt.xlabel('Fraction')
    plt.ylabel(f'Mean {options[i]}')
    plt.title(f'Mean {options[i]} vs. Fraction with Error Bars')

    # Add legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig(f'{PATH_SAVE}{options[i]}_basic.png')
    #plt.show()













# datasets = [
#     ("Architecture: Big, Epochs 30", signif_frac_big_30_processed[variant]),
#     ("Architecture: Big, Epochs 300", signif_frac_big_300_processed[variant]),
#     ("Architecture: Small, Epochs 30", signif_frac_small_30_processed[variant]),
#     ("Architecture: Small, Epochs 300", signif_frac_small_300_processed[variant])
# ]

# datasets = [
#     ("Architecture: 3 Layers, 512 Neurons, Epochs 30",  signif_frac_big_30_processed[variant] + signif_frac_big_30_augmented_processed[variant]),
#     ("Architecture: 1 Layer, 64 Neurons, Epochs 300", signif_frac_small_300_processed[variant] + signif_frac_small_300_augmented_processed[variant])
# ]

# mean_small_augment_only = np.mean(signif_frac_small_300_augmented_only_processed[variant])
# std_dev_small_augment_only = np.std(signif_frac_small_300_augmented_only_processed[variant])

# mean_big_augment_only = np.mean(signif_frac_big_30_augmented_only_processed[variant])
# std_dev_big_augment_only = np.std(signif_frac_big_30_augmented_only_processed[variant])




    # Define ranges for separate regions
    # region1 = np.arange(0.0, 1.1, 0.2)
    # region2 = np.arange(0.0, 1.1, 0.2)

    # if variant == 2:
    #     # Fill the area for region 1
    #     plt.fill_between(region1, 3.25, 4.7, color='green', alpha=0.3)
    #     plt.text(0.5, 3.7, 'Not augmented', horizontalalignment='center', verticalalignment='center', fontsize=12)


    #     # plt.text(4, 4, 'No augmentation', horizontalalignment='center', verticalalignment='center', fontsize=12)
    #     # Fill the area for region 2
    #     plt.fill_between(region2, 3.25, 4.7, color='blue', alpha=0.3)
    #     plt.text(1.5, 3.7, 'With augmentation', horizontalalignment='center', verticalalignment='center', fontsize=12)

    #     plt.axhline(y=mean_big_augment_only, color='red', linestyle='--', label=f'{options[variant]} without simulated data (Big)')
    #     plt.axhline(y=mean_small_augment_only, color='purple', linestyle='--', label=f'{options[variant]} without simulated data (Small)')