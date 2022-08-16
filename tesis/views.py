from django.shortcuts import render
from django.http import HttpRequest
from .models import enfermedadcrohn
from django.views.decorators.csrf import csrf_exempt

from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import render, redirect
from django.contrib.auth import logout as do_logout
from django.contrib.auth import authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as do_login
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpRequest
# Creacion de los PDF

from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.contrib.auth import logout as do_logout
from django.contrib.auth import authenticate
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login as do_login
from django.contrib.auth.hashers import make_password

#Añadimos el respectivo formulario con todas los features de la enfermedad de Crohn
@csrf_exempt
class FormularioAgregarfactorView(HttpRequest):
    def agregarfactor(request):
        if (request.method == "POST"):
            mienfermedad = enfermedadcrohn(numero_eventos=request.POST["numero_eventos"],
                                           imc=request.POST["imc"], altura=request.POST["altura"],
                                           sexo=request.POST["sexo"], edad=request.POST["edad"],
                                           peso=request.POST["peso"])
            d = enfermedadcrohn.objects.all()
            mienfermedad.save()
            mensaje = 'OK'
            datos = enfermedadcrohn.objects.last()
            # docente.objects.filter(Nombre=n).update(asignado=1)
            # print(request.POST)
            return render(request, "tesis/agregarfactor.html", {"datos": datos, "mensaje": mensaje})

        else:
            datos = enfermedadcrohn.objects.last()
            return render(request, "tesis/agregarfactor.html", {"datos": datos})


# Create your views here.
# def inicio(request):
#    return render(request, "tesis/inicio.html")
def editarfactor(request):
    if (request.method == "POST"):

        if (request.method == "POST"):
            tipo_tratamiento = enfermedadcrohn(tipo_tratamiento=request.POST["tipo_tratamiento"])
            d = enfermedadcrohn.objects.last()
            ide = d.id
            y = enfermedadcrohn.objects.filter(id=ide).first()

            enfermedadcrohn.objects.filter(id=ide).update(tipo_tratamiento=request.POST["tipo_tratamiento"])

            mensaje = 'OK'
            prediccion = 'Riesgo de Crohn'
            var = '1'
            ver = '1'

            datos = enfermedadcrohn.objects.last()
        return render(request, "tesis/agregarfactor.html",
                      {"datos": datos, 'Aprediccion': prediccion, "mensaje": mensaje, 'variable': var, 'ver': ver})

    else:
        datos = enfermedadcrohn.objects.last()
        return render(request, "tesis/agregarfactor.html", {"datos": datos})


# inicialmente
def inicio(request):
    if request.user.is_authenticated:
        return render(request, "tesis/inicio.html")
    else:
        return redirect('login')


def informacion(request):
    return render(request, "tesis/informacion.html")


def contactanos(request):
    return render(request, "tesis/contacto.html")


def logout(request):
    # Finalizamos la sesión
    do_logout(request)
    # Redireccionamos a la portada
    return redirect('/')


def login(request):
    # Creamos el formulario de autenticación vacío
    form = AuthenticationForm()
    if request.method == "POST":
        # Añadimos los datos recibidos al formulario
        form = AuthenticationForm(data=request.POST)
        # Si el formulario es válido...
        if form.is_valid():
            # Recuperamos las credenciales validadas
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            # Verificamos las credenciales del usuario
            user = authenticate(username=username, password=password)

            # Si existe un usuario con ese nombre y contraseña
            if user is not None:
                # Hacemos el login manualmente
                do_login(request, user)
                # Y le redireccionamos a la portada
                return redirect('inicio')

    # Si llegamos al final renderizamos el formulario
    return render(request, "tesis/login.html", {'form': form})


def resultadog(request, sklearn=None):
    import numpy as np
    import pandas as pd
    import sqlite3
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # Conexion a la base de datos
    db2 = sqlite3.connect("../db.sqlite3")

    filename = 'base_final.csv'
    db = pd.read_csv(filename, delimiter=",")
    db = db.replace(np.nan, "0")
    X = db[['numero_eventos', 'imc', 'altura', 'sexo', 'edad', 'peso']]  # seleccionar característica
    y = db[['resultado']]
    print("Arreglo de dimension y", y.shape)
    print("Datos de y", y.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #predicción del modelo Gradient Boosting Classification
    clff = GradientBoostingClassifier(learning_rate=0.005,
                                      n_estimators=1500,
                                      max_depth=3,
                                      min_samples_split=2,
                                      min_samples_leaf=1,
                                      max_features=6,
                                      subsample=1,
                                      random_state=100)
    clff.fit(X_train, y_train.values.ravel())
    precision = clff.score(X_test, y_test)
    print("Precision del Gradient:", precision)
    # print('Precision del modelo', precision)

    # p = clff.predict(X_test)
    # print('Predicción: %.3f' % yhat[0])

    # Para ver la estadisticas de la variable de la salida
    sqq = (
        "SELECT numero_eventos,imc,altura,sexo,edad,peso FROM tesis_enfermedadcrohn t order by id desc limit 1")

    preguntass = pd.read_sql_query(sqq, db2)

    Ss = preguntass[['numero_eventos', 'imc', 'altura', 'sexo', 'edad', 'peso']]

    dd = {'numero_eventos': [Ss.numero_eventos[0]], 'imc': [Ss.imc[0]], 'altura': [Ss.altura[0]],
          'sexo': [Ss.sexo[0]], 'edad': [Ss.edad[0]], 'peso': [Ss.peso[0]]}

    ww = pd.DataFrame(dd)
    ww = ww.replace(np.nan, "0")

    xtt = ww.iloc[:, [0, 1, 2, 3, 4, 5]].values

    pree = clff.predict(xtt)

    p = ''
    tt = ''
    # pree = int(pree)
    print(pree)
    if pree[0] == 1:

        p = 'Riesgo de Crohn'
        tt = 'placebo'

    elif pree[0] == 0:
        p = 'Sin Riesgo de Crohn'
        # tt = 'Farmaco'
    y_pred = clff.predict(X_test.values)
    print("Matriz Confusion Gradient", confusion_matrix(y_test, y_pred))
    data = 0, 86

    return render(request, 'tesis/agregarfactor.html', {'prediccion': p, 'precision': precision, 'tratamiento': tt,
                                                        'variable': pree[0]})

#Prediccion del modelo Árbol de Clasificación
def resultadoarbol(request, sklearn=None):
    import numpy as np
    import pandas as pd
    import sqlite3
    from sklearn.tree import DecisionTreeClassifier  # Import Arbol de Clasificación
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix

    # Conexion a la base de datos
    db2 = sqlite3.connect("../db.sqlite3")

    filename = 'base_final.csv'
    db = pd.read_csv(filename, delimiter=",")
    db = db.replace(np.nan, "0")
    X = db[['numero_eventos', 'imc', 'altura', 'sexo', 'edad', 'peso']]  # seleccionar característica
    y = db[['resultado']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clff = DecisionTreeClassifier(random_state=0)
    clff.fit(X, y)
    precision = clff.score(X_test, y_test)
    print("Precision del Arbol de Clasificación:", precision)

    # Para ver la estadisticas de la variable de la salida
    sqq = (
        "SELECT numero_eventos,imc,altura,sexo,edad,peso FROM tesis_enfermedadcrohn t order by id desc limit 1")

    preguntass = pd.read_sql_query(sqq, db2)

    Ss = preguntass[['numero_eventos', 'imc', 'altura', 'sexo', 'edad', 'peso']]

    dd = {'numero_eventos': [Ss.numero_eventos[0]], 'imc': [Ss.imc[0]], 'altura': [Ss.altura[0]],
          'sexo': [Ss.sexo[0]], 'edad': [Ss.edad[0]], 'peso': [Ss.peso[0]], }

    ww = pd.DataFrame(dd)
    ww = ww.replace(np.nan, "0")

    xtt = ww.iloc[:, [0, 1, 2, 3, 4, 5]].values

    pree = clff.predict(xtt)

    p = ''
    tt = ''
    # pree = int(pree)
    # print(pree)
    if pree[0] == 1:

        p = 'Riesgo de Crohn'
        tt = 'placebo'

    elif pree[0] == 0:
        p = 'Sin Riesgo de Crohn'
        # tt = 'Farmaco'
    y_pred = clff.predict(X_test)
    # print("f1 score Arbol", sklearn.metrics.f1_score(y_test, y_pred))
    print("Matriz Confusion Arbol", confusion_matrix(y_test,
                                                     y_pred))  # Pedimos la matriz de confusión de las predicciones del grupo Test. La diagonal de esta
    # matriz se lee: arriba a la izda True Negatives y abajo a la dcha True Positives.
    # cm = confusion_matrix(y_test, pree)

    return render(request, 'tesis/agregarfactor.html', {'prediccion2': p, 'precision2': precision, 'tratamiento2': tt})

#Por medio del método k-fold Cross-Validation elegimos el mejor modelo, obteniendo la media y desviación estándar de las métricas de exactitud,
# especificidad, Sensibilidad, F1 score .
import numpy as np
import pandas as pd
from numpy.core.numeric import indices
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, confusion_matrix, \
    f1_score, make_scorer

filename = 'base_final.csv'
registers = pd.read_csv(filename, delimiter=",")
castNumericRegisters = registers.replace(np.nan, "0")
X = castNumericRegisters[['numero_eventos', 'imc', 'altura', 'sexo', 'edad', 'peso']]
y = castNumericRegisters[['resultado']]

print(X.size)
print(y.size)
print(len(X.values.tolist()))


def toPorcentage(value):
    return value * 100


classifiers = [GradientBoostingClassifier(), DecisionTreeClassifier()]

indicators = {
    'accuracy': 'accuracy',
    'recall': 'recall',
    'f1': 'f1',
}

for classifier in classifiers:
    cross = cross_validate(estimator=classifier, X=X.to_numpy().tolist(), y=y.to_numpy().ravel(), cv=10,
                           scoring=indicators)

    print("Fit Time: ", toPorcentage(cross['fit_time']))
    print("Score Time: ", toPorcentage(cross['score_time']))
    print("Accuracy: ", cross['test_accuracy'])
    print("Mean Accuracy: ", toPorcentage(np.mean(cross['test_accuracy'])))
    print("Std Accuracy: ", np.std(cross['test_accuracy']))
    print("Recall: ", cross['test_recall'])
    print("Mean Recall: ", toPorcentage(np.mean(cross['test_recall'])))
    print("Std Recall: ", np.std(cross['test_recall']))
    print("F1: ", cross['test_f1'])
    print("Mean F1: ", toPorcentage(np.mean(cross['test_f1'])))
    print("Std F1: ", np.std(cross['test_f1']))

specificity = make_scorer(recall_score, pos_label=0)
spcificity_1 = cross_val_score(classifiers[0], X=X.to_numpy().tolist(), y=y.to_numpy().ravel(), cv=15,
                               scoring=specificity)
spcificity_2 = cross_val_score(classifiers[1], X=X.to_numpy().tolist(), y=y.to_numpy().ravel(), cv=15,
                               scoring=specificity)

print(spcificity_1)
print("Mean specificity 1: ", toPorcentage(np.mean(spcificity_1)))
print("Std specificity 1: ", np.std(spcificity_1))

print(spcificity_2)
print("Mean specificity 2: ", toPorcentage(np.mean(spcificity_2)))
print("Std specificity 2: ", np.std(spcificity_2))


#Por medio de la división del dataset (entrenamiento y validación) seguimos evaluando el mejor modelo predictivo
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import classification_report, precision_score, accuracy_score, recall_score, confusion_matrix, \
    f1_score

filename = 'base_final.csv'
registers = pd.read_csv(filename, delimiter=",")
castNumericRegisters = registers.replace(np.nan, "0")
X = castNumericRegisters[['numero_eventos', 'imc', 'altura', 'sexo', 'edad', 'peso']]
y = castNumericRegisters[['resultado']]

# Primer algoritmo GradientBoostingClassifier()

gradientBoost = GradientBoostingClassifier(
    learning_rate=0.005,
    n_estimators=1500,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=6,
    subsample=1,
    random_state=100
)

# y_pred = cross_val_predict(estimator=gradientBoost, X=X.to_numpy().tolist(), y=y.to_numpy().ravel(), cv=2)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, random_state=100)
gradientBoost.fit(X_train_1, y_train_1.values.ravel())

y_predict_1 = gradientBoost.predict(X_test_1)

tn_1, fp_1, fn_1, tp_1 = confusion_matrix(y_test_1, y_predict_1).ravel()

specificity_boost = tn_1 / (tn_1 + fp_1)  # Calcular especificidad usando matriz de confusion

print('Exactitud: ', accuracy_score(y_test_1, y_predict_1))
print('Sensibilidad (Recall):', recall_score(y_test_1, y_predict_1))
print('Especificidad: ', specificity_boost)
print('F1 score: ', f1_score(y_test_1, y_predict_1))

print('Negativos reales_1: ', tn_1)
print('Falsos positivos_1: ', fp_1)
print('Falsos negativos_1: ', fn_1)
print('Positivos reales_1: ', tp_1)

print('Tasa falsos negativos 1: ', fn_1 / (fn_1 + tp_1))
print('Tasa falsos positivos 1: ', fp_1 / (tn_1 + fp_1))

print(classification_report(y_test_1, y_predict_1))

# Segundo algoritmo DecisionTreeClassifier()

decisionTree = DecisionTreeClassifier()

# y_pred = cross_val_predict(estimator=decisionTree, X=X.to_numpy().tolist(), y=y.to_numpy().ravel(), cv=2)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, random_state=0)
decisionTree.fit(X_train_2, y_train_2.values.ravel())

y_predict_2 = decisionTree.predict(X_test_2)

tn_2, fp_2, fn_2, tp_2 = confusion_matrix(y_test_2, y_predict_2).ravel()

specificity_tree = tn_2 / (tn_2 + fp_2)  # Calcular especificidad usando matriz de confusion

print('Exactitud: ', accuracy_score(y_test_2, y_predict_2))
print('Sensibilidad (Recall):', recall_score(y_test_2, y_predict_2))
print('Especificidad: ', specificity_tree)
print('F1 score: ', f1_score(y_test_2, y_predict_2))

print('Negativos reales_2: ', tn_2)
print('Falsos positivos_2: ', fp_2)
print('Falsos negativos_2: ', fn_2)
print('Positivos reales_2: ', tp_2)

print('Tasa falsos negativos 1: ', fn_2 / (fn_2 + tp_2))
print('Tasa falsos positivos 1: ', fp_2 / (tn_2 + fp_2))

print(classification_report(y_test_2, y_predict_2))


#Aplicamos el tunning al mejor modelo seleccionado (Gradient Boosting Classification) y determinamos los parámetros definidos.
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Read csv file
filename = 'base_final.csv'
registers = pd.read_csv(filename, delimiter=",")
castNumericRegisters = registers.replace(np.nan, "0")

# Read Features
X = castNumericRegisters[['numero_eventos', 'imc', 'altura', 'sexo', 'edad', 'peso']]
y = castNumericRegisters[['resultado']]

# New instance of classifier
gradientBoost = GradientBoostingClassifier(
    learning_rate=0.005,
    n_estimators=1500,
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=6,
    subsample=1,
    random_state=100
)

# Split our data: 70% fit and 30% test and shuffle it 5 times
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=100)

# We use our 70% of data to fit using gradient boost algorithm
gradientBoost.fit(x_train_1, y_train_1.values.ravel())
result = gradientBoost.predict(x_test_1)

print(classification_report(y_test_1, result))

# Se trabaja con los valores por defecto, y se pone el random state


primer_ajuste_hyperparametro = {
    'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
    'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750]
}
# tunning_1 = GridSearchCV(estimator=GradientBoostingClassifier(random_state=10),
#                        param_grid=primer_ajuste_hyperparametro,
#                        scoring='accuracy',
#                        n_jobs=4,
#                        cv=5)
# tunning_1.fit(x_train_1, y_train_1.values.ravel())
# print(tunning_1.cv_results_, tunning_1.best_params_, tunning_1.best_score_)

segundo_ajuste_hyperparametro = {'max_depth': [2, 3, 4, 5, 6, 7]}
# tunning_2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.005, n_estimators=1500, random_state=10),
#                         param_grid=segundo_ajuste_hyperparametro,
#                         scoring='accuracy',
#                         n_jobs=4,
#                         cv=5)
# tunning_2.fit(x_train_1, y_train_1.values.ravel())
# print(tunning_2.cv_results_, tunning_2.best_params_, tunning_2.best_score_)

tercer_ajuste_hyperparametro = {
    'min_samples_split': [2, 4, 6, 8, 10, 20, 40, 60, 100],
    'min_samples_leaf': [1, 3, 5, 7, 9]
}
# tunning_3 = GridSearchCV(estimator=GradientBoostingClassifier(max_depth=3, learning_rate=0.005, n_estimators=1500,
#                                                              random_state=10),
#                         param_grid=tercer_ajuste_hyperparametro,
#                         scoring='accuracy',
#                         n_jobs=4,
#                         cv=5)
# tunning_3.fit(x_train_1, y_train_1.values.ravel())
# print(tunning_3.cv_results_, tunning_3.best_params_, tunning_3.best_score_)

cuarto_ajuste_hyperparametro = {'max_features': [2, 3, 4, 5, 6, 7]}
# tunning_4 = GridSearchCV(estimator=GradientBoostingClassifier(min_samples_leaf=1, min_samples_split=2, max_depth=3,
#                                                              learning_rate=0.005, n_estimators=1500,
#                                                              random_state=10),
#                         param_grid=cuarto_ajuste_hyperparametro,
#                         scoring='accuracy',
#                         n_jobs=4,
#                         cv=5)
# tunning_4.fit(x_train_1, y_train_1.values.ravel())
# print(tunning_4.cv_results_, tunning_4.best_params_, tunning_4.best_score_)

quinto_ajuste_hyperparametro = {'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]}
# tunning_5 = GridSearchCV(estimator=GradientBoostingClassifier(min_samples_leaf=1, min_samples_split=2, max_features=6,
#                                                              max_depth=3, learning_rate=0.005, n_estimators=1500,
#                                                              random_state=10),
#                         param_grid=quinto_ajuste_hyperparametro,
#                         scoring='accuracy',
#                         n_jobs=4,
#                         cv=5)
# tunning_5.fit(x_train_1, y_train_1.values.ravel())
# print(tunning_5.cv_results_, tunning_5.best_params_, tunning_5.best_score_)
