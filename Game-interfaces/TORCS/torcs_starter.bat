:: Batch file to run torcs with some settings
:: First parameter (%1) is xml config file
:: Second parameter (%2) is path to java classes with torcs client
:: Parameter %3 is path to directory with AI internal script
:: %4 is path to torcs installation directory
:: %5 is full path of python.exe

cd %4
start /b wtorcs.exe -r %1 -t 1000000 -nofuel -nodamage -nolaptime
::start /b wtorcs.exe -r ./config/raceman/race_config.xml -t 1000000 -nofuel -nodamage -nolaptime
::timeout 2 /nobreak
PING 1.1.1.1 -n 1 -w 1000 > NUL
java -cp %2 scr.Client scr.GeneralAIDriver port:3002 torcs:%3 python:%5
