:: Batch file to run torcs with some settings; parameters:
:: %1 is xml config file
:: %2 is path to java classes with torcs client
:: %3 port
:: %4 is full path to torcs installation directory

cd %4
start /b wtorcs.exe %1 -t 1000000 -nofuel -nolaptime > NUL
::start /b wtorcs.exe -r ./config/raceman/race_config.xml -t 1000000 -nofuel -nodamage -nolaptime
::timeout 2 /nobreak
PING 1.1.1.1 -n 1 -w 300 > NUL
echo java -cp %2 scr.Client scr.GeneralAIDriver port:%3
java -cp %2 scr.Client scr.GeneralAIDriver port:%3