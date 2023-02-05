FROM ubuntu:22.04

# Ubuntu no longer distributes chromium-browser outside of snap
#
# Proposed solution: https://askubuntu.com/questions/1204571/how-to-install-chromium-without-snap

# Add debian buster
RUN echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster.gpg] http://deb.debian.org/debian buster main\n\
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-buster-updates.gpg] http://deb.debian.org/debian buster-updates main\n\
deb [arch=amd64 signed-by=/usr/share/keyrings/debian-security-buster.gpg] http://deb.debian.org/debian-security buster/updates main\n'\
>> /etc/apt/sources.list.d/debian.list

# Add keys
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys DCC9EFBF77E11517 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 648ACFD622F3D138 \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 112695A0E562B32A \
    && apt-key export 77E11517 | gpg --dearmour -o /usr/share/keyrings/debian-buster.gpg \
    && apt-key export 22F3D138 | gpg --dearmour -o /usr/share/keyrings/debian-buster-updates.gpg \
    && apt-key export E562B32A | gpg --dearmour -o /usr/share/keyrings/debian-security-buster.gpg

# Prefer debian repo for chromium* packages only
# Note the double-blank lines between entries
RUN <<EOF cat >> /etc/apt/preferences.d/chromium.pref
RUN echo 'Package: *\n\
Pin: release a=eoan\n\
Pin-Priority: 500\n\
\n\
\n\
Package: *\n\
Pin: origin "deb.debian.org"\n\
Pin-Priority: 300\n\
\n\
\n\
Package: chromium*\n\
Pin: origin "deb.debian.org"\n\
Pin-Priority: 700\n'\
>> /etc/apt/preferences.d/chromium.pref

RUN apt-get install -y chromium chromium-drive

RUN pip3 install -r requirements.txt

COPY . .
