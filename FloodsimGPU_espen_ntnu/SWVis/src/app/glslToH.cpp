/*****************************************************************************/
/*                                                                           */
/*                                                                           */
/* (c) Copyright 2010, 2011, 2012 by                                         */
/*     SINTEF, Oslo, Norway                                                  */
/*     All rights reserved.                                                  */
/*                                                                           */
/*  THIS SOFTWARE IS FURNISHED UNDER A LICENSE AND MAY BE USED AND COPIED    */
/*  ONLY IN  ACCORDANCE WITH  THE  TERMS  OF  SUCH  LICENSE  AND WITH THE    */
/*  INCLUSION OF THE ABOVE COPYRIGHT NOTICE. THIS SOFTWARE OR  ANY  OTHER    */
/*  COPIES THEREOF MAY NOT BE PROVIDED OR OTHERWISE MADE AVAILABLE TO ANY    */
/*  OTHER PERSON.  NO TITLE TO AND OWNERSHIP OF  THE  SOFTWARE IS  HEREBY    */
/*  TRANSFERRED.                                                             */
/*                                                                           */
/*  SINTEF  MAKES NO WARRANTY  OF  ANY KIND WITH REGARD TO THIS SOFTWARE,    */
/*  INCLUDING,   BUT   NOT   LIMITED   TO,  THE  IMPLIED   WARRANTIES  OF    */
/*  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.                    */
/*                                                                           */
/*  SINTEF SHALL NOT BE  LIABLE  FOR  ERRORS  CONTAINED HEREIN OR DIRECT,    */
/*  SPECIAL,  INCIDENTAL  OR  CONSEQUENTIAL  DAMAGES  IN  CONNECTION WITH    */
/*  FURNISHING, PERFORMANCE, OR USE OF THIS MATERIAL.                        */
/*                                                                           */
/*                                                                           */
/*****************************************************************************/

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <boost/xpressive/xpressive.hpp>

using namespace std;
using namespace boost::xpressive;




string regex_escape(string text){
	const sregex re_escape_text = sregex::compile("([\\\"\\\\])");
	const sregex re_remove_linefeed = sregex::compile("([\\n\\r])");
    text = regex_replace( text, re_escape_text, string("\\$1") );
    text = regex_replace( text, re_remove_linefeed, string("") );
    return text;
}


/**
 * This app simple takes each line of an input glsl-file, and writes it out as a
 * header file, suitable for compilation into an app.
 */
int main(int argc, char** argv) {
	char* input_file;
	char* output_file;
	ifstream input;
	ofstream output;
	string line;

	if (argc != 3) {
		cout << "Usage: " << argv[0] << " <shaderfile> <outputfile>" << endl;
		exit(-1);
	}
	else {
		input_file = argv[1];
		output_file = argv[2];
		if (strcmp(input_file, output_file) == 0) {
			cout << "Cannot use the same input and output file." << endl;
			cout << "Input was " << input_file << ", " << endl;
			cout << "Output was " << output_file << "." << endl;
			exit(-1);
		}
		else {
			cout << "Reading from " << input_file << ", writing to " << output_file << "." << endl;
		}
	}

	input.open(input_file);
	if(!input.good()) {
		cout << "Error opening " << input_file << " for reading." << endl;
		exit(-1);
	}

	output.open(output_file);
	if (!output.good()) {
		cout << "Error opening " << output_file << " for writing." << endl;
		exit(-1);
	}

	output << "#include <string>" << endl;
	output << "static std::string source = \"\\" << endl;
	while(getline(input, line)) {
		output << regex_escape(line) << "\\n\\" << endl;
	}
	output << "\";" << endl;

	return 0;
}
