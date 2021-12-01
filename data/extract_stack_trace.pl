#!/usr/local/bin/perl
use warnings;
use utf8;
use Try::Tiny;

use Parse::StackTrace;
use JSON -convert_blessed_universally;


#use open ':std', ':encoding(utf8)';
use open qw(:std :utf8);

my $in = join("", <STDIN>);
my $json = JSON->new->utf8->allow_nonref->convert_blessed;
my @obj = @{$json->decode($in)};
my @list = ();
my $trace;
my $js;

foreach my $o (@obj){
    try{
        $trace =  Parse::StackTrace->parse(types => ['GDB', 'Python'],
                                         text => $o,
                                         debug => 0);
    }catch{
        return ".Something wrong has happened.";
    };


    if ($trace){
#        $js =$json->encode($trace);
        push(@list, $trace);
    }else{
#        push(@list, '{}');
        push(@list, undef);
    }
}
print($json->encode(\@list));


