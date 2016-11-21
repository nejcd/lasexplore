import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import andrews_curves

matplotlib.style.use('ggplot')


#Filepath
path = 'S:\Dropbox\dev\Data/'
filename = '46_all_class'


columns = ['linearity', 'planarity', 'scattering', 'omnivariance', 'anisotropy', 'eigentropy', 'classification', 'intensity']
df = pd.read_csv(path + filename + 'cvs')
df.columns = columns

print df.head()
print 'Data types:'
print df.dtypes


#get classification
classes = df.classification.unique()

# for clas in classes:
#     print 'Stats for classification {0}'.format(clas)
#     dc = df[df.classification == clas]
#     print dc.describe()
#     dc.plot.box(title=clas)




#Plot Andrew curves
plt.figure()
andrews_curves(df, 'classification', alpha=0.4)


plt.show()


