#!/bin/bash

rsync -zarv -e "ssh -i parallel.pem" --exclude='.*' --exclude='cmake*' --include="*/" --include="*".{cpp,h,c,txt,py} --include="Makefile" --include="job-*" --include="auto-*" --include="login-*" --exclude="*" ../$1 ubuntu@34.215.219.109:~