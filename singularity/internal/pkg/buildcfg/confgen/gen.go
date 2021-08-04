// Copyright (c) 2018-2019, Sylabs Inc. All rights reserved.
// This software is licensed under a 3-clause BSD license. Please consult the
// LICENSE.md file distributed with the sources of this project regarding your
// rights to use or distribute this software.

package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"regexp"
	"strings"
	"text/template"
)

func parseLine(s string) (d Define) {
	d = Define{
		Words: strings.Fields(s),
	}

	return
}

// Define is a struct that contains one line of configuration words.
type Define struct {
	Words []string
}

// WriteLine writes a line of configuration.
func (d Define) WriteLine(isSuidInstall bool) (s string) {
	s = d.Words[2]
	if len(d.Words) > 3 {
		for _, w := range d.Words[3:] {
			s += " + " + w
		}
	}

	varType := "const"
	varStatement := d.Words[1] + " = " + s

	// For security, always mark variables as const when building with SetUID support
	if !isSuidInstall {
		// Apply runtime relocation to some variables
		switch d.Words[1] {
		case
			"BINDIR",
			"LIBEXECDIR",
			"SYSCONFDIR",
			"SESSIONDIR",
			"SINGULARITY_CONFDIR",
			"PLUGIN_ROOTDIR":
			varType = "var"
			varStatement = d.Words[1] + " = RelocatePath(" + s + ")"
		default:
			// Some variables are defined relative to others and cannot be const
			if strings.Contains(s, "SINGULARITY_CONFDIR") {
				varType = "var"
			}
		}
	}

	return varType + " " + varStatement
}

var confgenTemplate = template.Must(template.New("").Parse(`// Code generated by go generate; DO NOT EDIT.
package buildcfg
{{if not .IsSuidInstall}}
import (
    "os"
	"path/filepath"
	"strings"
)

func RelocatePath(original string) (string) {
	// For security, never allow relocation when built with SetUID support
	if SINGULARITY_SUID_INSTALL != 0 {
		panic("This code should not exist when SINGULARITY_SUID_INSTALL is set")
	}

	if ! strings.HasPrefix(original, "{{.Prefix}}") {
		return original
	}

	executablePath, err := os.Executable()
	if err != nil {
		return original
	}
	prefix := filepath.Dir(executablePath)

	switch filepath.Base(executablePath) {
	case "singularity":
		// PREFIX/bin/singularity
		prefix = filepath.Dir(prefix)
	case "starter":
		// PREFIX/libexec/singularity/bin/starter
		prefix = filepath.Dir(filepath.Dir(filepath.Dir(prefix)))
	default:
		return original
	}

	relativePath, err := filepath.Rel("{{.Prefix}}", original)
	if err != nil {
		panic(err)
	}

	result := filepath.Join(prefix, relativePath)
	return result
}
{{end}}

{{ range $i, $d := .Defines }}
{{$d.WriteLine $.IsSuidInstall -}}
{{end}}
`))

func main() {
	outFile, err := os.Create("config.go")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer outFile.Close()

	// Determin if this is a setuid install
	b, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Println(err)
		return
	}
	re := regexp.MustCompile("#define SINGULARITY_SUID_INSTALL ([01])")
	isSuidInstall := true
	switch re.FindStringSubmatch(string(b))[1] {
	case "0":
		isSuidInstall = false
	case "1":
		isSuidInstall = true
	default:
		panic("Failed to parse value of SINGULARITY_SUID_INSTALL")
	}

	// Parse the config.h file
	inFile, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Println(err)
		return
	}

	header := []Define{}
	s := bufio.NewScanner(bytes.NewReader(inFile))
	prefix := ""
	for s.Scan() {
		d := parseLine(s.Text())
		if len(d.Words) > 2 && d.Words[0] == "#define" {
			if d.Words[1] == "PREFIX" {
				if len(d.Words) != 3 {
					panic("Expected PREFIX to contain 3 elements")
				}
				prefix = d.Words[2]
			}
			header = append(header, d)
		}
	}
	if prefix == "" {
		panic("Failed to find value of PREFIX")
	}

	if goBuildTags := os.Getenv("GO_BUILD_TAGS"); goBuildTags != "" {
		d := Define{
			Words: []string{
				"#define",
				"GO_BUILD_TAGS",
				fmt.Sprintf("`%s`", goBuildTags),
			},
		}
		header = append(header, d)
	}

	data := struct {
		Prefix        string
		Defines       []Define
		IsSuidInstall bool
	}{
		prefix[1 : len(prefix)-1],
		header,
		isSuidInstall,
	}
	err = confgenTemplate.Execute(outFile, data)
	if err != nil {
		fmt.Println(err)
		return
	}
}
