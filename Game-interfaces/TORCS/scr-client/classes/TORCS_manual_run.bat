:: Run from current directory. TORCS server has to run before this batch file.
:: 1. parameter is a model file to run

set mypath=%cd%
java -cp ".;../lib/*" scr.Client scr.GeneralAIDriver ai_script:%mypath%"/../../../../Controller/script.py" python:"C:\\Program Files\\Anaconda3\\envs\\tensorflow-gpu\\python.exe" model_config:%1