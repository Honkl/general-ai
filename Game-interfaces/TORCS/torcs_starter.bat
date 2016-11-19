:: Batch file to run torcs with some settings; parameters:
:: %1 is xml config file
:: %2 is path to java classes with torcs client
:: %3 port
:: %4 is full path of AI internal script
:: %5 is full path to torcs installation directory
:: %6 is full path of python.exe
:: %7 is a full path of model configuration file

cd %5
start /b wtorcs.exe -r %1 -t 1000000 -nofuel -nodamage -nolaptime
::start /b wtorcs.exe -r ./config/raceman/race_config.xml -t 1000000 -nofuel -nodamage -nolaptime
::timeout 2 /nobreak
PING 1.1.1.1 -n 1 -w 1000 > NUL
echo java -cp %2 scr.Client scr.GeneralAIDriver port:%3 ai_script:%4 python:%6 model_config:%7
java -cp %2 scr.Client scr.GeneralAIDriver port:%3 ai_script:%4 python:%6 model_config:%7
