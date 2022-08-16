# CrohnDisease
## Pasos para ejecutar programa
- Abrir una sesión de terminal o shell
- Localizarse en la raiz del proyecto
- Se debe configurar ambiente virtual, a continuación link de guía: **[Guía de ambientes virtuales](https://codigonaranja.com/como-trabajar-con-ambientes-virtuales-en-python "Guía de ambientes virtuales")**
- Instalar la dependencia virtualenv
```shell
pip install virtualenv
```
- Para crear un ambiente virtual ejecutar lo siguiente:
```shell
virtualenv -p python3 venv
```
- Activar virtualenv
```shell
venv\Scripts\activate
```
- A continuación se verá algo así en la terminal
```shell
(venv) Directorio Proyecto>
```
- Ejecutar el siguiente comando para instalar las dependencias
```shell
pip install -r requirements.txt
```
- Para ejecutar el programa se debe escribir el siguiente comando:
```shell
./manage.py runserver 8000
```
- Luego deberá abrir el navegador de su preferencia y escribir la siguiente url: [http://127.0.0.1:8000](http://127.0.0.1:8000 "http://127.0.0.1:8000")