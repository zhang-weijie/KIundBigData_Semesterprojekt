## KIundBigData - Entwicklung eines neuronalen Netzwerkmodells zur Diagnose von Lungenentzündungen
Dieses Projekt konzentriert sich auf die Entwicklung eines neuronalen Netzwerkmodells zur automatischen Diagnose von Lungenentzündungen anhand von Brust-Röntgenbildern. Ziel ist es, die Genauigkeit der Diagnosen zu verbessern und die Belastung für das medizinische Personal zu reduzieren.

### Projektübersicht
#### Zielstellung: 
Entwicklung eines neuronalen Netzwerks zur Diagnose von Lungenentzündungen.
#### Datensatz: 
5.863 Röntgenbilder in zwei Kategorien (Pneumonie/Normal), aufgeteilt in drei Ordner (Train, Test, Val). https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
#### Analyse und Visualisierung: 
Voranalyse des Datensatzes, Überprüfung von Bilderverteilung, Klassenbalance, Bildgröße und -qualität.
#### Datenbereinigung und -transformation: 
Entfernung von Ausreißern, Kontrastverstärkung, Rauschreduzierung.
#### Modellwahl und Leistungsbewertung: 
Einsatz eines Convolutional Neural Network (CNN) und Evaluierung mittels Kreuzvalidierung.
#### Verbesserungsansätze: 
Anwendung von Methoden gegen Overfitting wie L2-Regularisierung und Dropout.

### Implementierungsdetails
#### Sprache und Bibliotheken: 
Python, PyTorch, PIL, Matplotlib, Seaborn, Scikit-Learn.
#### Modell: 
Einfaches CNN mit Schichten für Konvolution, Pooling und Fully-Connected Layer.
#### Regularisierung: 
L2-Regularisierung und Dropout zur Vorbeugung von Overfitting.
#### Datenverarbeitung: 
Transformationen wie Grayscale-Konvertierung, Bildgrößenanpassung, Kontrastverstärkung und Gaußsche Filterung.
#### Trainingsstrategie: 
k-Fold Cross-Validation und alternativ traditionelle Aufteilung in Trainings-, Validierungs- und Testdaten.
