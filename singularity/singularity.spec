# 
# Copyright (c) 2017-2018, SyLabs, Inc. All rights reserved.
# Copyright (c) 2017, SingularityWare, LLC. All rights reserved.
#
# Copyright (c) 2015-2017, Gregory M. Kurtzer. All rights reserved.
# 
# Copyright (c) 2016, The Regents of the University of California, through
# Lawrence Berkeley National Laboratory (subject to receipt of any required
# approvals from the U.S. Dept. of Energy).  All rights reserved.
# 
# This software is licensed under a customized 3-clause BSD license.  Please
# consult LICENSE file distributed with the sources of this project regarding
# your rights to use or distribute this software.
# 
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so. 
# 
# 


%{!?_rel:%{expand:%%global _rel 1}}

Summary: Application and environment virtualization
Name: singularity
Version: 2.5.0
Release: %{_rel}%{?dist}
# https://spdx.org/licenses/BSD-3-Clause-LBNL.html
License: BSD-3-Clause-LBNL
Group: System Environment/Base
URL: http://singularity.lbl.gov/
Source: %{name}-%{version}.tar.gz
ExclusiveOS: linux
BuildRoot: %{?_tmppath}%{!?_tmppath:/var/tmp}/%{name}-%{version}-%{release}-root
BuildRequires: python
BuildRequires: libarchive-devel
%if "%{_target_vendor}" == "suse"
Requires: squashfs
%else
Requires: squashfs-tools
%endif

Requires: %{name}-runtime = %{version}-%{release}

%description
Singularity provides functionality to make portable
containers that can be used across host environments.

%package devel
Summary: Development libraries for Singularity
Group: System Environment/Development

%description devel
Development files for Singularity

%package runtime
Summary: Support for running Singularity containers
Group: System Environment/Base

%description runtime
This package contains support for running containers created
by the %{name} package.

%prep
%setup


%build
if [ ! -f configure ]; then
  ./autogen.sh
fi

%configure
%{__make} %{?mflags}


%install
%{__make} install DESTDIR=$RPM_BUILD_ROOT %{?mflags_install}
rm -f $RPM_BUILD_ROOT/%{_libdir}/singularity/lib*.la

%post runtime -p /sbin/ldconfig
%postun runtime -p /sbin/ldconfig

%clean
rm -rf $RPM_BUILD_ROOT

%files
%doc examples CONTRIBUTORS.md CONTRIBUTING.md COPYRIGHT.md INSTALL.md LICENSE-LBNL.md LICENSE.md README.md
%attr(0755, root, root) %dir %{_sysconfdir}/singularity
%attr(0644, root, root) %config(noreplace) %{_sysconfdir}/singularity/*

%{_libexecdir}/singularity/cli/apps.*
%{_libexecdir}/singularity/cli/bootstrap.*
%{_libexecdir}/singularity/cli/build.*
%{_libexecdir}/singularity/cli/check.*
%{_libexecdir}/singularity/cli/create.*
%{_libexecdir}/singularity/cli/image.*
%{_libexecdir}/singularity/cli/inspect.*
%{_libexecdir}/singularity/cli/mount.*
%{_libexecdir}/singularity/cli/pull.*
%{_libexecdir}/singularity/cli/selftest.*
%{_libexecdir}/singularity/helpers
%{_libexecdir}/singularity/python

# Binaries
%{_libexecdir}/singularity/bin/builddef
%{_libexecdir}/singularity/bin/cleanupd
%{_libexecdir}/singularity/bin/get-section
%{_libexecdir}/singularity/bin/mount
%{_libexecdir}/singularity/bin/image-type
%{_libexecdir}/singularity/bin/prepheader
%{_libexecdir}/singularity/bin/docker-extract

# Directories
%{_libexecdir}/singularity/bootstrap-scripts

#SUID programs
%attr(4755, root, root) %{_libexecdir}/singularity/bin/mount-suid

%files runtime
%dir %{_libexecdir}/singularity
%dir %{_localstatedir}/singularity
%dir %{_localstatedir}/singularity/mnt
%dir %{_localstatedir}/singularity/mnt/session
%dir %{_localstatedir}/singularity/mnt/container
%dir %{_localstatedir}/singularity/mnt/overlay
%dir %{_localstatedir}/singularity/mnt/final
%{_bindir}/singularity
%{_bindir}/run-singularity
%{_libdir}/singularity/lib*.so.*
%{_libexecdir}/singularity/cli/action_argparser.*
%{_libexecdir}/singularity/cli/exec.*
%{_libexecdir}/singularity/cli/help.*
%{_libexecdir}/singularity/cli/instance.*
%{_libexecdir}/singularity/cli/run.*
%{_libexecdir}/singularity/cli/shell.*
%{_libexecdir}/singularity/cli/test.*
%{_libexecdir}/singularity/bin/action
%{_libexecdir}/singularity/bin/start
%{_libexecdir}/singularity/bin/docker-extract
%{_libexecdir}/singularity/functions
%{_libexecdir}/singularity/handlers
%{_libexecdir}/singularity/image-handler.sh
%dir %{_sysconfdir}/singularity
%config(noreplace) %{_sysconfdir}/singularity/*
%{_mandir}/man1/singularity.1*
%dir %{_sysconfdir}/bash_completion.d
%{_sysconfdir}/bash_completion.d/singularity

#SUID programs
%attr(4755, root, root) %{_libexecdir}/singularity/bin/action-suid
%attr(4755, root, root) %{_libexecdir}/singularity/bin/start-suid

%files devel
%defattr(-, root, root)
%{_libdir}/singularity/lib*.so
%{_libdir}/singularity/lib*.a
%{_includedir}/singularity/*.h


%changelog

