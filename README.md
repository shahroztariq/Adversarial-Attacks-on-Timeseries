## Towards an Awareness of Time Series Anomaly Detection Models’ Adversarial Vulnerability

### Overview

#### Pipeline of a typical time series training phase and adversarial attack phase
<img src="https://i.ibb.co/WfLfgMS/Adv-ICLR-Pipeline.png" alt="Coffee" border="0">

<hr/>

#### Example of ground truth vs perturbed time series using FGSM and PGD attacks on CLMPPCA

<img src="https://i.ibb.co/NW9CtM1/Realvs-Adv.png" alt="Coffee" border="0" width="500" ><img src="https://i.ibb.co/pJ7mfmX/Realvs-Adv-error.png" alt="Coffee" border="0" width="500" >

<hr/>

#### FGSM and PGD attacks on MSCRED (left) and  MTAD-GAT (right)

<img src="https://i.ibb.co/Sm4SpFy/MSCRED.png" alt="Coffee" border="0" width="500" ><img src="https://i.ibb.co/j5LMBYB/MTAD.png" alt="Coffee" border="0" width="500" >

<hr/>
<!---
<img src="https://i.ibb.co/87Fxy61/MSCRED-3.png" alt="Coffee" border="0" width="500" >
-->

## Additional Experiments on UCR Dataset
In addition to all the experiment on state-of-the-art anomaly and intrusion detection system. We also cover general time series classification task where we attack a multilayer perception (MLP), a fully convolutional network and ResNet trained on different dataset from the UCR repository. We conduct an analysis of 71 datasets from the University of California, Riverside (UCR) repository. In future work, we will expand on this experiment by including additional neural networks (MobileNet, EfficientNet, DenseNet, and Inception Time) and datasets (the remainder of the UCR dataset, datasets from the UEA repository). 

We find that the Carlini-Wagner L2 attack provides the best adversarial examples, resulting in the most significant performance degradation. In Figure below, we show some original samples and the corresponding perturbed samples generated by FGSM, PGD, BIM, Carlini-Wagner L2, and MIM attacks on UCR datasets. Also, in Tables below, we present the classification results for MLP, FCN, and ResNet.

 <img src="https://i.ibb.co/bRWfBNJ/Adiac-attacks-2.png" alt="Adiac" border="0" width="500"><img src="https://i.ibb.co/QKnpsZh/Car-attacks-1.png" alt="Car" border="0" width="500" > 
<img src="https://i.ibb.co/f1cKd6z/FISH-attacks-0.png" alt="FISH" border="0" width="500"><img src="https://i.ibb.co/KDzGtwW/Meat-attacks-2.png" alt="Meat" border="0" width="500">
<img src="https://i.ibb.co/SyR00sK/Coffee-attacks-1.png" alt="Coffee" border="0" width="500"><img src="https://i.ibb.co/RCHbRQJ/Diatom-Size-Reduction-attacks-0.png" alt="Diatom-Size-Reduction" border="0" width="500">

---

## Adversarial attacks on an MLP trained on different UCR datasets 
| Datasets                             | FGSM     | PGD     | BIM     | Madry et al. 2016 | Carlini <br> Wagner <br> L2 | MIM     | No <br> Attack |
| :----------------------------------- | -------: | ------: | ------: | ----------------: | --------------------------: | ------: | -------------: |
| 50words                              | 44±0\.8  | 42±1\.3 | 42±1\.3 | 42±1\.3           | 35±1                        | 43±1    | 63±1\.1        |
| Adiac                                | 14±1\.8  | 15±1\.6 | 15±1\.6 | 15±1\.6           | 16±1\.3                     | 16±1\.8 | 53±2\.7        |
| ArrowHead                            | 29±3\.9  | 27±3\.3 | 27±3\.3 | 27±3\.3           | 24±4\.3                     | 27±3\.1 | 74±2\.6        |
| Beef                                 | 32±5\.1  | 26±3\.9 | 26±3\.9 | 26±3\.9           | 29±3\.9                     | 27±3\.4 | 78±3\.9        |
| BeetleFly                            | 74±7\.7  | 74±7\.7 | 74±7\.7 | 74±7\.7           | 70±5                        | 74±7\.7 | 75±13\.3       |
| BirdChicken                          | 62±5\.8  | 62±5\.8 | 62±5\.8 | 62±5\.8           | 57±10\.5                    | 62±5\.8 | 69±2\.9        |
| Car                                  | 49±1     | 35±2\.9 | 35±2\.9 | 35±2\.9           | 52±1                        | 45±1    | 83±1           |
| CBF                                  | 76±2\.6  | 76±2\.4 | 76±2\.4 | 76±2\.4           | 63±3\.5                     | 76±2\.6 | 94±2\.6        |
| Chlorine <br> Concentration          | 24±0\.3  | 24±0\.5 | 24±0\.5 | 24±0\.5           | 24±0\.7                     | 24±0\.4 | 65±0\.4        |
| Coffee                               | 9±2\.1   | 9±2\.1  | 9±2\.1  | 9±2\.1            | 9±4\.2                      | 9±2\.1  | 100±0          |
| Computers                            | 46±1\.1  | 45±1\.1 | 45±1\.1 | 45±1\.1           | 45±1\.1                     | 45±1\.1 | 58±0\.9        |
| Cricket\_X                           | 26±0\.7  | 25±0\.7 | 25±0\.7 | 25±0\.7           | 21±1                        | 26±0\.9 | 45±1           |
| Cricket\_Y                           | 30±0\.8  | 29±1\.7 | 29±1\.7 | 29±1\.7           | 24±0\.6                     | 29±1\.7 | 48±1\.6        |
| Cricket\_Z                           | 32±0\.7  | 30±1\.1 | 30±1\.1 | 30±1\.1           | 25±1                        | 31±0\.2 | 44±1\.2        |
| DiatomSize <br> Reduction            | 40±1\.2  | 37±1\.4 | 37±1\.4 | 37±1\.4           | 31±4                        | 38±1\.5 | 95±2\.4        |
| DistalPhalanx <br> OutlineAgeGroup   | 16±0\.9  | 16±1    | 16±1    | 16±1              | 16±1                        | 16±0\.9 | 83±0\.8        |
| DistalPhalanx <br> OutlineCorrect    | 29±1\.4  | 28±1\.8 | 28±1\.8 | 28±1\.8           | 25±0\.9                     | 29±1\.7 | 77±0\.9        |
| Distal <br> PhalanxTW                | 13±0\.8  | 12±1\.1 | 12±1\.1 | 12±1\.1           | 12±0\.9                     | 12±1\.2 | 78±0\.7        |
| Earthquakes                          | 69±1\.5  | 69±1\.5 | 69±1\.5 | 69±1\.5           | 52±4\.2                     | 69±1\.5 | 73±1\.1        |
| ECG200                               | 60±1\.8  | 60±2\.1 | 60±2\.1 | 60±2\.1           | 29±5\.8                     | 60±2\.1 | 84±0\.6        |
| ECG5000                              | 65±0\.2  | 64±0\.3 | 64±0\.3 | 64±0\.3           | 61±0\.3                     | 64±0\.4 | 93±0\.2        |
| ECGFiveDays                          | 48±2\.3  | 46±2\.1 | 46±2\.1 | 46±2\.1           | 35±4\.8                     | 47±2\.1 | 95±3\.3        |
| ElectricDevices                      | 22±0\.4  | 21±0\.5 | 21±0\.5 | 21±0\.5           | 21±0\.6                     | 21±0\.6 | 55±0\.8        |
| FaceAll                              | 57±0\.3  | 56±0\.4 | 56±0\.4 | 56±0\.4           | 39±0\.9                     | 56±0\.2 | 74±0\.6        |
| FaceFour                             | 79±2\.4  | 77±2    | 77±2    | 77±2              | 76±1\.8                     | 79±1\.4 | 88±0\.7        |
| FacesUCR                             | 67±1\.7  | 63±1\.6 | 63±1\.6 | 63±1\.6           | 55±1\.6                     | 65±1\.8 | 83±1\.2        |
| FISH                                 | 16±2\.1  | 8±1\.2  | 8±1\.2  | 8±1\.2            | 14±1\.9                     | 12±1\.2 | 85±0\.4        |
| Gun\_Point                           | 48±6\.1  | 47±6\.2 | 47±6\.2 | 47±6\.2           | 34±5\.4                     | 47±6\.2 | 92±1\.4        |
| Ham                                  | 34±2\.4  | 34±2\.6 | 34±2\.6 | 34±2\.6           | 48±3\.5                     | 34±2\.6 | 70±2           |
| Haptics                              | 21±0\.9  | 21±0\.8 | 21±0\.8 | 21±0\.8           | 21±1\.2                     | 20±0\.7 | 41±0\.7        |
| Herring                              | 50±1\.9  | 50±1\.9 | 50±1\.9 | 50±1\.9           | 50±1\.9                     | 50±1\.9 | 51±1\.9        |
| InlineSkate                          | 21±1\.1  | 19±0\.8 | 19±0\.8 | 19±0\.8           | 20±1\.4                     | 20±1\.4 | 34±0\.7        |
| InsectWingbeat <br> Sound            | 37±0\.7  | 30±0\.3 | 30±0\.3 | 30±0\.3           | 42±0\.3                     | 34±0\.4 | 62±0\.7        |
| ItalyPower <br> Demand               | 82±0\.8  | 82±0\.9 | 82±0\.9 | 82±0\.9           | 11±1\.4                     | 82±0\.9 | 96±0\.2        |
| LargeKitchen <br> Appliances         | 33±2\.2  | 32±1\.3 | 32±1\.3 | 32±1\.3           | 34±0\.6                     | 33±2\.1 | 51±0\.5        |
| Lighting2                            | 70±2\.6  | 70±2\.6 | 70±2\.6 | 70±2\.6           | 58±3\.8                     | 70±2\.6 | 65±3\.5        |
| Lighting7                            | 53±4\.2  | 53±3\.7 | 53±3\.7 | 53±3\.7           | 35±3\.7                     | 53±3\.7 | 64±2\.4        |
| Meat                                 | 26±1     | 26±1    | 26±1    | 26±1              | 25±1\.7                     | 26±1    | 74±1           |
| MedicalImages                        | 39±1\.9  | 36±2\.2 | 36±2\.2 | 36±2\.2           | 26±0\.5                     | 37±2\.2 | 67±0\.5        |
| MiddlePhalanx <br> OutlineAgeGroup   | 32±10\.7 | 26±4\.8 | 26±4\.8 | 26±4\.8           | 20±0\.8                     | 27±5\.7 | 73±1\.5        |
| MiddlePhalanx <br> OutlineCorrect    | 46±1\.5  | 46±1\.6 | 46±1\.6 | 46±1\.6           | 45±1\.5                     | 46±1\.6 | 56±1\.5        |
| Middle <br> PhalanxTW                | 18±2\.9  | 18±2\.8 | 18±2\.8 | 18±2\.8           | 18±1\.7                     | 18±2\.9 | 56±2\.4        |
| MoteStrain                           | 79±0\.7  | 79±0\.7 | 79±0\.7 | 79±0\.7           | 53±2\.3                     | 79±0\.7 | 84±1\.1        |
| OliveOil                             | 28±2     | 28±2    | 28±2    | 28±2              | 28±2                        | 28±2    | 59±2           |
| OSULeaf                              | 29±0\.7  | 29±1\.1 | 29±1\.1 | 29±1\.1           | 29±0\.9                     | 30±0\.7 | 45±0\.3        |
| Phalanges <br> OutlinesCorrect       | 33±3\.2  | 33±2\.6 | 33±2\.6 | 33±2\.6           | 33±2\.3                     | 33±2\.7 | 68±2\.4        |
| Plane                                | 89±2     | 87±1\.1 | 87±1\.1 | 87±1\.1           | 85±4\.3                     | 88±1\.1 | 98±1\.1        |
| ProximalPhalanx <br> OutlineAgeGroup | 18±2     | 18±2\.3 | 18±2\.3 | 18±2\.3           | 18±1\.8                     | 18±2\.3 | 81±1\.9        |
| ProximalPhalanx <br> OutlineCorrect  | 36±1\.4  | 34±1\.1 | 34±1\.1 | 34±1\.1           | 33±1\.6                     | 34±0\.9 | 68±1\.6        |
| Proximal <br> PhalanxTW              | 41±3\.9  | 42±4    | 42±4    | 42±4              | 42±4                        | 42±3\.9 | 53±4\.1        |
| Refrigeration <br> Devices           | 36±1\.8  | 36±1\.6 | 36±1\.6 | 36±1\.6           | 36±1\.3                     | 36±1\.9 | 43±1\.2        |
| ScreenType                           | 39±1\.4  | 38±1\.8 | 38±1\.8 | 38±1\.8           | 38±1                        | 39±1\.6 | 36±0\.3        |
| ShapeletSim                          | 50±1\.7  | 50±1\.4 | 50±1\.4 | 50±1\.4           | 49±1\.7                     | 50±1\.4 | 48±0\.9        |
| ShapesAll                            | 49±1\.6  | 42±1\.1 | 42±1\.1 | 42±1\.1           | 43±1\.3                     | 46±1\.8 | 70±0\.2        |
| SmallKitchen <br> Appliances         | 33±1\.4  | 34±1    | 34±1    | 34±1              | 36±1\.6                     | 34±1\.1 | 49±2\.2        |
| SonyAIBO <br> RobotSurface           | 68±2\.6  | 68±2\.6 | 68±2\.6 | 68±2\.6           | 62±7\.3                     | 68±2\.6 | 68±4\.6        |
| SonyAIBO <br> RobotSurfaceII         | 81±0\.8  | 81±0\.8 | 81±0\.8 | 81±0\.8           | 71±0\.6                     | 81±0\.8 | 83±0\.8        |
| Strawberry                           | 7±0\.3   | 6±0\.3  | 6±0\.3  | 6±0\.3            | 9±0\.7                      | 7±0\.2  | 96±0\.3        |
| SwedishLeaf                          | 32±1\.2  | 26±2\.1 | 26±2\.1 | 26±2\.1           | 25±0\.8                     | 29±1\.4 | 82±0\.3        |
| Symbols                              | 76±1\.5  | 74±1\.2 | 74±1\.2 | 74±1\.2           | 76±1\.4                     | 75±1    | 89±0\.2        |
| synthetic\_control                   | 80±1\.6  | 80±1\.7 | 80±1\.7 | 80±1\.7           | 37±3\.6                     | 80±1\.6 | 95±1           |
| ToeSegmentation1                     | 51±1\.5  | 51±1\.5 | 51±1\.5 | 51±1\.5           | 50±1\.2                     | 51±1\.5 | 57±0\.7        |
| ToeSegmentation2                     | 63±1\.8  | 63±1\.8 | 63±1\.8 | 63±1\.8           | 55±5\.5                     | 63±1\.8 | 67±3           |
| Trace                                | 29±2\.7  | 29±2\.4 | 29±2\.4 | 29±2\.4           | 29±2\.4                     | 29±2\.9 | 89±1\.8        |
| TwoLeadECG                           | 45±2\.2  | 44±2\.3 | 44±2\.3 | 44±2\.3           | 37±1\.8                     | 45±2\.2 | 77±0\.7        |
| Two\_Patterns                        | 32±1\.8  | 31±1\.6 | 31±1\.6 | 31±1\.6           | 12±0\.2                     | 31±1\.7 | 96±0\.4        |
| wafer                                | 39±1\.5  | 39±1\.5 | 39±1\.5 | 39±1\.5           | 21±1\.5                     | 39±1\.5 | 96±0\.9        |
| Wine                                 | 45±0     | 45±0    | 45±0    | 45±0              | 45±0                        | 45±0    | 56±0           |
| WordsSynonyms                        | 40±1\.2  | 38±0\.5 | 38±0\.5 | 38±0\.5           | 32±1                        | 39±1\.1 | 53±0\.4        |
| Worms                                | 28±0\.4  | 27±0\.9 | 27±0\.9 | 27±0\.9           | 24±1\.5                     | 28±0\.6 | 36±1\.2        |
| WormsTwoClass                        | 49±1\.2  | 49±1    | 49±1    | 49±1              | 47±1\.4                     | 49±1    | 60±1           |

---

## Adversarial attacks on a FCN trained on different UCR datasets 
| Datasets                             | FGSM     | PGD      | BIM      | Madry et al. 2016 | Carlini Wagner L2 | MIM      | No Attack |
| :----------------------------------- | -------: | -------: | -------: | ----------------: | ----------------: | -------: | --------: |
| 50words                              | 3±0\.5   | 6±1\.4   | 6±1\.4   | 6±1\.4            | 18±3\.6           | 4±1\.3   | 29±16     |
| Adiac                                | 5±1\.8   | 7±3\.8   | 7±3\.8   | 7±3\.8            | 11±2\.1           | 7±3\.5   | 24±17\.7  |
| ArrowHead                            | 40±0     | 14±6\.2  | 14±6\.2  | 14±6\.2           | 14±6\.5           | 15±6     | 80±6\.6   |
| Beef                                 | 26±10\.2 | 23±9\.7  | 23±9\.7  | 23±9\.7           | 23±12\.7          | 22±7\.7  | 52±9\.7   |
| BeetleFly                            | 50±0     | 20±5     | 20±5     | 20±5              | 20±5              | 20±5     | 80±5      |
| BirdChicken                          | 50±0     | 15±10    | 15±10    | 15±10             | 7±2\.9            | 22±2\.9  | 94±2\.9   |
| Car                                  | 22±0     | 40±27\.5 | 40±27\.5 | 40±27\.5          | 40±26\.2          | 40±25\.1 | 47±23\.4  |
| CBF                                  | 83±1\.2  | 79±1\.6  | 79±1\.6  | 79±1\.6           | 1±0\.1            | 81±1\.3  | 100±0\.2  |
| Chlorine <br> Concentration          | 39±19\.5 | 39±19\.8 | 39±19\.8 | 39±19\.8          | 38±19\.1          | 39±19\.8 | 54±18\.5  |
| Coffee                               | 0±0      | 0±0      | 0±0      | 0±0               | 0±0               | 0±0      | 100±0     |
| Computers                            | 44±10    | 19±5\.7  | 19±5\.7  | 19±5\.7           | 16±6\.1           | 28±11    | 85±6\.1   |
| Cricket\_X                           | 16±5\.7  | 11±1\.8  | 11±1\.8  | 11±1\.8           | 13±2\.3           | 11±3     | 72±3\.7   |
| Cricket\_Y                           | 19±1\.9  | 16±3\.1  | 16±3\.1  | 16±3\.1           | 16±2\.9           | 16±3\.3  | 69±7\.5   |
| Cricket\_Z                           | 13±1\.1  | 11±3\.2  | 11±3\.2  | 11±3\.2           | 14±3\.5           | 11±2\.1  | 72±5\.1   |
| DiatomSize <br> Reduction            | 16±4\.9  | 6±0\.9   | 6±0\.9   | 6±0\.9            | 7±0\.5            | 7±0\.7   | 93±0\.7   |
| DistalPhalanx <br> OutlineAgeGroup   | 19±4\.7  | 19±4\.4  | 19±4\.4  | 19±4\.4           | 19±4\.4           | 19±4\.4  | 80±4\.3   |
| DistalPhalanx <br> OutlineCorrect    | 38±9\.6  | 32±6\.1  | 32±6\.1  | 32±6\.1           | 32±6\.2           | 33±6\.6  | 69±6\.1   |
| Distal <br> PhalanxTW                | 15±1\.1  | 17±1\.2  | 17±1\.2  | 17±1\.2           | 17±1\.1           | 17±1\.1  | 73±2\.1   |
| Earthquakes                          | 36±4\.1  | 34±3\.2  | 34±3\.2  | 34±3\.2           | 25±2\.5           | 35±3\.3  | 76±2\.5   |
| ECG200                               | 49±6\.5  | 16±3\.1  | 16±3\.1  | 16±3\.1           | 11±1\.8           | 24±5     | 89±1\.8   |
| ECG5000                              | 69±6\.9  | 33±24\.7 | 33±24\.7 | 33±24\.7          | 4±0\.4            | 51±12\.5 | 94±0\.4   |
| ECGFiveDays                          | 38±9\.5  | 2±0\.2   | 2±0\.2   | 2±0\.2            | 2±0\.3            | 2±0\.3   | 99±0\.3   |
| ElectricDevices                      | 43±1\.3  | 32±2\.7  | 32±2\.7  | 32±2\.7           | 14±3\.3           | 35±2\.9  | 70±3\.7   |
| FaceAll                              | 66±0\.7  | 41±0\.4  | 41±0\.4  | 41±0\.4           | 8±2\.7            | 57±0\.4  | 90±2\.8   |
| FaceFour                             | 6±2\.3   | 3±1\.8   | 3±1\.8   | 3±1\.8            | 5±1\.8            | 3±1\.2   | 94±0\.7   |
| FacesUCR                             | 68±2\.4  | 40±7\.9  | 40±7\.9  | 40±7\.9           | 4±0\.7            | 56±4\.4  | 93±0\.8   |
| FISH                                 | 13±0\.4  | 19±11\.5 | 19±11\.5 | 19±11\.5          | 22±11\.9          | 18±11    | 60±2\.9   |
| Gun\_Point                           | 51±2\.7  | 2±0\.7   | 2±0\.7   | 2±0\.7            | 1±0\.4            | 4±2\.4   | 100±0\.4  |
| Ham                                  | 37±3\.4  | 37±3\.5  | 37±3\.5  | 37±3\.5           | 37±3\.5           | 37±3\.5  | 64±3\.5   |
| Haptics                              | 23±3\.1  | 18±4\.8  | 18±4\.8  | 18±4\.8           | 19±5              | 18±4\.8  | 29±3\.4   |
| Herring                              | 60±0     | 46±8\.2  | 46±8\.2  | 46±8\.2           | 49±11\.9          | 54±5\.5  | 60±0      |
| InlineSkate                          | 16±0\.5  | 13±5\.2  | 13±5\.2  | 13±5\.2           | 16±6\.7           | 13±4\.5  | 22±7\.6   |
| InsectWingbeat <br> Sound            | 13±1\.8  | 11±1\.3  | 11±1\.3  | 11±1\.3           | 12±1\.5           | 11±1\.4  | 23±4\.4   |
| ItalyPower <br> Demand               | 84±1     | 81±1\.7  | 81±1\.7  | 81±1\.7           | 5±0\.5            | 83±1\.5  | 96±0\.3   |
| LargeKitchen <br> Appliances         | 50±4\.9  | 32±23\.7 | 32±23\.7 | 32±23\.7          | 21±17\.5          | 45±13\.9 | 74±16     |
| Lighting2                            | 40±1\.7  | 29±1     | 29±1     | 29±1              | 29±1              | 30±1\.7  | 72±1      |
| Lighting7                            | 32±7\.6  | 19±2\.9  | 19±2\.9  | 19±2\.9           | 17±3\.5           | 23±4\.2  | 74±1\.6   |
| Meat                                 | 34±0     | 45±13\.7 | 45±13\.7 | 45±13\.7          | 52±24\.9          | 47±11\.7 | 34±0      |
| MedicalImages                        | 23±6\.8  | 14±2     | 14±2     | 14±2              | 14±3\.1           | 16±1\.2  | 77±2\.8   |
| MiddlePhalanx <br> OutlineAgeGroup   | 18±6\.6  | 18±5\.9  | 18±5\.9  | 18±5\.9           | 17±5\.7           | 18±6\.1  | 70±6\.7   |
| MiddlePhalanx <br> OutlineCorrect    | 44±22\.5 | 43±21\.6 | 43±21\.6 | 43±21\.6          | 45±24\.2          | 43±21\.6 | 58±21\.4  |
| Middle <br> PhalanxTW                | 20±10    | 23±11    | 23±11    | 23±11             | 21±9              | 23±10\.7 | 48±12\.8  |
| MoteStrain                           | 80±1     | 78±1\.2  | 78±1\.2  | 78±1\.2           | 10±0\.5           | 79±1\.5  | 91±0\.5   |
| OliveOil                             | 18±19\.3 | 16±21\.2 | 16±21\.2 | 16±21\.2          | 18±19\.3          | 18±19\.3 | 56±15\.1  |
| OSULeaf                              | 14±0     | 12±4     | 12±4     | 12±4              | 12±4\.4           | 11±4\.1  | 75±16\.7  |
| Phalanges <br> OutlinesCorrect       | 36±2\.5  | 36±2\.5  | 36±2\.5  | 36±2\.5           | 36±2\.6           | 36±2\.5  | 65±2\.6   |
| Plane                                | 40±5\.8  | 11±3\.9  | 11±3\.9  | 11±3\.9           | 0±0               | 25±6\.5  | 100±0     |
| ProximalPhalanx <br> OutlineAgeGroup | 32±23\.7 | 22±8\.8  | 22±8\.8  | 22±8\.8           | 25±10\.7          | 22±8\.8  | 64±18\.9  |
| ProximalPhalanx <br> OutlineCorrect  | 32±26\.8 | 31±26\.4 | 31±26\.4 | 31±26\.4          | 31±26\.2          | 31±26\.8 | 70±26\.2  |
| Proximal <br> PhalanxTW              | 18±8\.2  | 14±3\.1  | 14±3\.1  | 14±3\.1           | 15±4\.7           | 14±2\.9  | 75±2\.9   |
| Refrigeration <br> Devices           | 40±3\.5  | 36±0\.9  | 36±0\.9  | 36±0\.9           | 35±1\.7           | 36±1     | 46±1\.7   |
| ScreenType                           | 33±3\.3  | 28±3\.6  | 28±3\.6  | 28±3\.6           | 27±3\.6           | 29±4\.3  | 62±5\.2   |
| ShapeletSim                          | 8±3\.7   | 8±3\.1   | 8±3\.1   | 8±3\.1            | 8±2\.8            | 8±3\.1   | 93±2\.8   |
| ShapesAll                            | 4±1\.4   | 3±2\.9   | 3±2\.9   | 3±2\.9            | 7±0\.6            | 3±1\.9   | 19±18     |
| SmallKitchen <br> Appliances         | 53±16\.7 | 37±18\.1 | 37±18\.1 | 37±18\.1          | 39±22\.6          | 41±11\.1 | 43±12\.3  |
| SonyAIBO <br> RobotSurface           | 84±2\.2  | 82±2\.7  | 82±2\.7  | 82±2\.7           | 5±0\.3            | 83±2\.7  | 97±0\.6   |
| SonyAIBO <br> RobotSurfaceII         | 86±1\.5  | 84±2\.1  | 84±2\.1  | 84±2\.1           | 3±0\.5            | 85±1\.7  | 98±0\.5   |
| Strawberry                           | 44±20\.8 | 31±8\.8  | 31±8\.8  | 31±8\.8           | 31±8\.9           | 31±9\.1  | 70±8\.8   |
| SwedishLeaf                          | 28±1\.7  | 10±2\.6  | 10±2\.6  | 10±2\.6           | 6±3\.6            | 13±3\.3  | 93±3\.6   |
| Symbols                              | 36±3\.2  | 6±1\.6   | 6±1\.6   | 6±1\.6            | 5±0\.6            | 15±1\.9  | 94±1\.3   |
| synthetic\_control                   | 95±1     | 95±1\.3  | 95±1\.3  | 95±1\.3           | 3±0\.9            | 95±1\.2  | 98±0\.7   |
| ToeSegmentation1                     | 41±6\.2  | 11±0\.8  | 11±0\.8  | 11±0\.8           | 3±0\.7            | 18±3     | 98±0\.7   |
| ToeSegmentation2                     | 43±1\.4  | 26±2\.3  | 26±2\.3  | 26±2\.3           | 14±2\.8           | 36±0\.5  | 87±2\.8   |
| Trace                                | 52±18\.6 | 18±8\.9  | 18±8\.9  | 18±8\.9           | 1±0\.6            | 43±2\.9  | 100±0\.6  |
| TwoLeadECG                           | 7±3\.1   | 2±0\.4   | 2±0\.4   | 2±0\.4            | 1±0\.1            | 3±0\.7   | 100±0\.1  |
| Two\_Patterns                        | 34±7\.3  | 15±0\.7  | 15±0\.7  | 15±0\.7           | 15±0\.7           | 19±2\.3  | 86±0\.7   |
| wafer                                | 8±3\.2   | 3±0\.9   | 3±0\.9   | 3±0\.9            | 1±0\.2            | 3±1\.3   | 100±0\.2  |
| Wine                                 | 50±0     | 50±0     | 50±0     | 50±0              | 50±0              | 50±0     | 50±0      |
| WordsSynonyms                        | 5±2\.2   | 9±3\.3   | 9±3\.3   | 9±3\.3            | 12±1\.5           | 6±1\.9   | 30±10\.2  |
| Worms                                | 17±1\.7  | 21±3\.6  | 21±3\.6  | 21±3\.6           | 21±5\.3           | 21±3\.4  | 48±7\.3   |
| WormsTwoClass                        | 48±5     | 39±2\.3  | 39±2\.3  | 39±2\.3           | 39±2\.5           | 40±4\.2  | 62±2\.3   |

---

## Adversarial attacks on ResNet trained on different UCR datasets 
| Datasets                             | FGSM     | PGD      | BIM      | Madry et al. 2016 | Carlini Wagner L2 | MIM      | No Attack |
| :----------------------------------- | -------: | -------: | -------: | ----------------: | ----------------: | -------: | --------: |
| 50words                              | 8±2\.3   | 10±1     | 10±1     | 10±1              | 13±1\.5           | 9±1\.5   | 67±0\.7   |
| Adiac                                | 5±0\.2   | 10±1\.2  | 10±1\.2  | 10±1\.2           | 10±0\.2           | 10±0\.4  | 82±0\.7   |
| ArrowHead                            | 34±11\.5 | 13±0\.9  | 13±0\.9  | 13±0\.9           | 13±1\.5           | 15±1     | 79±2\.3   |
| Beef                                 | 24±8\.9  | 19±5\.1  | 19±5\.1  | 19±5\.1           | 18±3\.9           | 22±3\.9  | 74±3\.4   |
| BeetleFly                            | 29±5\.8  | 17±5\.8  | 17±5\.8  | 17±5\.8           | 17±5\.8           | 17±5\.8  | 84±5\.8   |
| BirdChicken                          | 54±5\.8  | 14±2\.9  | 14±2\.9  | 14±2\.9           | 14±2\.9           | 20±5     | 87±2\.9   |
| Car                                  | 20±1     | 9±4\.5   | 9±4\.5   | 9±4\.5            | 8±3\.9            | 10±4\.9  | 89±3\.5   |
| CBF                                  | 89±1\.4  | 87±1\.8  | 87±1\.8  | 87±1\.8           | 1±0\.2            | 88±1\.6  | 100±0\.2  |
| Chlorine <br> Concentration          | 14±0\.4  | 14±0\.8  | 14±0\.8  | 14±0\.8           | 13±0\.4           | 14±0\.7  | 82±1\.1   |
| Coffee                               | 0±0      | 0±0      | 0±0      | 0±0               | 0±0               | 0±0      | 100±0     |
| Computers                            | 58±5\.4  | 24±1\.3  | 24±1\.3  | 24±1\.3           | 20±3\.2           | 45±5\.1  | 82±2\.6   |
| Cricket\_X                           | 33±3     | 17±2\.5  | 17±2\.5  | 17±2\.5           | 14±2\.1           | 27±1\.9  | 76±2\.4   |
| Cricket\_Y                           | 23±0\.6  | 13±0\.7  | 13±0\.7  | 13±0\.7           | 13±0\.6           | 16±1\.7  | 80±1\.1   |
| Cricket\_Z                           | 28±2\.9  | 14±2     | 14±2     | 14±2              | 13±0\.8           | 22±2\.4  | 78±1\.4   |
| DiatomSize <br> Reduction            | 10±4\.1  | 4±1\.5   | 4±1\.5   | 4±1\.5            | 5±2               | 4±1\.4   | 97±1\.9   |
| DistalPhalanx <br> OutlineAgeGroup   | 18±2\.4  | 17±1\.8  | 17±1\.8  | 17±1\.8           | 17±2              | 17±1\.8  | 81±1\.8   |
| DistalPhalanx <br> OutlineCorrect    | 29±3\.6  | 23±1     | 23±1     | 23±1              | 21±1\.2           | 25±1\.7  | 80±1      |
| Distal <br> PhalanxTW                | 15±0\.3  | 15±0\.8  | 15±0\.8  | 15±0\.8           | 14±0\.6           | 15±0\.9  | 76±0\.7   |
| Earthquakes                          | 48±2\.9  | 45±2\.7  | 45±2\.7  | 45±2\.7           | 24±1              | 46±3\.1  | 80±1\.2   |
| ECG200                               | 69±4\.4  | 50±11\.6 | 50±11\.6 | 50±11\.6          | 13±2\.1           | 63±4\.1  | 88±2\.4   |
| ECG5000                              | 73±0\.8  | 61±1\.3  | 61±1\.3  | 61±1\.3           | 5±0\.3            | 66±1\.3  | 94±0\.3   |
| ECGFiveDays                          | 33±16\.2 | 4±1\.6   | 4±1\.6   | 4±1\.6            | 3±0\.6            | 6±3\.8   | 98±0\.7   |
| ElectricDevices                      | 41±2\.1  | 31±1\.7  | 31±1\.7  | 31±1\.7           | 15±2\.4           | 36±2\.2  | 70±4\.5   |
| FaceAll                              | 76±0\.4  | 69±1     | 69±1     | 69±1              | 11±0\.5           | 74±0\.7  | 83±1\.6   |
| FaceFour                             | 30±5\.2  | 9±2\.4   | 9±2\.4   | 9±2\.4            | 4±2\.9            | 22±3\.5  | 95±0\.7   |
| FacesUCR                             | 74±1\.4  | 64±2\.3  | 64±2\.3  | 64±2\.3           | 3±0\.8            | 70±1\.6  | 95±0\.4   |
| FISH                                 | 13±0     | 3±0\.9   | 3±0\.9   | 3±0\.9            | 3±1\.2            | 3±0\.9   | 98±1      |
| Gun\_Point                           | 23±5\.6  | 6±2      | 6±2      | 6±2               | 1±0\.4            | 10±0\.7  | 100±0     |
| Ham                                  | 30±2\.9  | 29±2     | 29±2     | 29±2              | 30±2\.4           | 29±2     | 72±2      |
| Haptics                              | 20±0\.2  | 22±3\.1  | 22±3\.1  | 22±3\.1           | 21±3\.7           | 21±3\.8  | 49±4      |
| Herring                              | 49±11    | 41±1     | 41±1     | 41±1              | 41±1              | 41±1     | 60±1      |
| InlineSkate                          | 15±1\.3  | 19±2     | 19±2     | 19±2              | 19±2\.9           | 19±1\.9  | 32±3\.1   |
| InsectWingbeat <br> Sound            | 22±0\.5  | 23±0\.6  | 23±0\.6  | 23±0\.6           | 23±0\.4           | 24±0\.3  | 46±1\.1   |
| ItalyPower <br> Demand               | 87±1\.3  | 86±0\.8  | 86±0\.8  | 86±0\.8           | 7±0\.9            | 86±1\.3  | 97±0\.2   |
| LargeKitchen <br> Appliances         | 59±2\.8  | 32±2\.7  | 32±2\.7  | 32±2\.7           | 8±1\.4            | 47±1\.2  | 90±0\.8   |
| Lighting2                            | 46±0     | 42±2\.6  | 42±2\.6  | 42±2\.6           | 27±1\.7           | 43±1\.7  | 74±1\.7   |
| Lighting7                            | 36±3\.7  | 20±4\.2  | 20±4\.2  | 20±4\.2           | 19±2\.1           | 24±7\.7  | 74±4\.2   |
| Meat                                 | 17±15\.5 | 8±5\.4   | 8±5\.4   | 8±5\.4            | 8±5\.4            | 8±5\.4   | 93±5\.4   |
| MedicalImages                        | 47±5     | 28±3\.8  | 28±3\.8  | 28±3\.8           | 15±2\.5           | 36±2\.4  | 78±0\.7   |
| MiddlePhalanx <br> OutlineAgeGroup   | 16±1\.5  | 16±0\.7  | 16±0\.7  | 16±0\.7           | 15±0\.2           | 16±0\.8  | 75±1      |
| MiddlePhalanx <br> OutlineCorrect    | 27±9\.1  | 27±9     | 27±9     | 27±9              | 27±9\.1           | 27±9     | 74±9\.2   |
| Middle <br> PhalanxTW                | 15±2\.8  | 17±0\.4  | 17±0\.4  | 17±0\.4           | 17±0\.6           | 17±0\.7  | 62±0\.8   |
| MoteStrain                           | 76±0\.9  | 73±1\.1  | 73±1\.1  | 73±1\.1           | 10±0\.8           | 75±1\.1  | 91±0\.8   |
| OliveOil                             | 14±0     | 17±5\.8  | 17±5\.8  | 17±5\.8           | 18±3\.9           | 17±5\.8  | 79±2      |
| OSULeaf                              | 14±1     | 6±2\.2   | 6±2\.2   | 6±2\.2            | 5±1\.9            | 6±2\.2   | 94±2\.8   |
| Phalanges <br> OutlinesCorrect       | 27±3     | 17±0\.9  | 17±0\.9  | 17±0\.9           | 18±0\.7           | 17±0\.9  | 84±0\.9   |
| Plane                                | 73±6\.2  | 41±6\.4  | 41±6\.4  | 41±6\.4           | 0±0               | 63±5\.3  | 100±0     |
| ProximalPhalanx <br> OutlineAgeGroup | 16±4\.8  | 15±0\.8  | 15±0\.8  | 15±0\.8           | 16±1\.5           | 15±0\.8  | 86±0\.6   |
| ProximalPhalanx <br> OutlineCorrect  | 16±2\.6  | 11±1\.6  | 11±1\.6  | 11±1\.6           | 11±1\.7           | 11±1\.6  | 90±1\.6   |
| Proximal <br> PhalanxTW              | 8±1\.2   | 13±0\.5  | 13±0\.5  | 13±0\.5           | 14±0\.3           | 14±0\.4  | 82±0\.5   |
| Refrigeration <br> Devices           | 35±2\.5  | 34±3\.1  | 34±3\.1  | 34±3\.1           | 31±2\.3           | 34±3\.1  | 54±0\.6   |
| ScreenType                           | 35±7     | 29±2\.6  | 29±2\.6  | 29±2\.6           | 28±3\.5           | 32±4\.5  | 61±3\.8   |
| ShapeletSim                          | 13±7\.9  | 12±8\.6  | 12±8\.6  | 12±8\.6           | 10±10\.2          | 13±8\.1  | 91±9\.9   |
| ShapesAll                            | 7±0\.7   | 3±0\.3   | 3±0\.3   | 3±0\.3            | 5±0\.3            | 4±0\.7   | 88±0\.5   |
| SmallKitchen <br> Appliances         | 44±4\.5  | 28±5\.6  | 28±5\.6  | 28±5\.6           | 29±7\.7           | 34±5\.2  | 56±16     |
| SonyAIBO <br> RobotSurface           | 80±2\.3  | 79±2\.9  | 79±2\.9  | 79±2\.9           | 14±3\.2           | 79±2\.5  | 92±0\.9   |
| SonyAIBO <br> RobotSurfaceII         | 81±1\.1  | 79±1\.6  | 79±1\.6  | 79±1\.6           | 4±0\.8            | 80±1     | 98±0\.8   |
| Strawberry                           | 24±16    | 22±17\.7 | 22±17\.7 | 22±17\.7          | 22±17\.6          | 22±17\.7 | 80±17\.6  |
| SwedishLeaf                          | 34±0\.8  | 16±0\.5  | 16±0\.5  | 16±0\.5           | 4±0\.5            | 22±0\.9  | 96±0\.4   |
| Symbols                              | 32±2\.1  | 8±0\.5   | 8±0\.5   | 8±0\.5            | 5±1\.6            | 16±1\.5  | 95±1\.7   |
| synthetic\_control                   | 95±0\.7  | 95±0\.4  | 95±0\.4  | 95±0\.4           | 20±4              | 95±0\.7  | 100±0\.4  |
| ToeSegmentation1                     | 54±1\.8  | 31±2\.5  | 31±2\.5  | 31±2\.5           | 4±0\.7            | 39±2     | 97±0\.7   |
| ToeSegmentation2                     | 45±5\.2  | 35±5\.9  | 35±5\.9  | 35±5\.9           | 11±2\.5           | 41±4\.3  | 90±2\.5   |
| Trace                                | 30±2\.1  | 13±9\.7  | 13±9\.7  | 13±9\.7           | 2±1\.6            | 37±8\.6  | 98±0      |
| TwoLeadECG                           | 8±4\.8   | 2±0\.6   | 2±0\.6   | 2±0\.6            | 1±0\.5            | 4±1\.7   | 100±0\.3  |
| Two\_Patterns                        | 68±1\.9  | 42±6\.2  | 42±6\.2  | 42±6\.2           | 6±1\.1            | 56±3\.8  | 96±1      |
| wafer                                | 17±11\.8 | 7±7\.8   | 7±7\.8   | 7±7\.8            | 2±0\.2            | 11±10\.8 | 100±0\.1  |
| Wine                                 | 34±16    | 25±8\.4  | 25±8\.4  | 25±8\.4           | 25±8\.4           | 25±8\.4  | 76±8\.4   |
| WordsSynonyms                        | 15±3\.1  | 14±1     | 14±1     | 14±1              | 16±0\.4           | 14±1\.4  | 54±1\.3   |
| Worms                                | 26±2     | 21±1\.5  | 21±1\.5  | 21±1\.5           | 19±0\.9           | 25±0\.4  | 63±2      |
| WormsTwoClass                        | 54±2\.7  | 29±2     | 29±2     | 29±2              | 27±2              | 32±1\.4  | 75±1\.4   |
