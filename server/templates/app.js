

       $(function() {
    $('#predict').click(function() {
        event.preventDefault();
        var form_data = new FormData($('#myform')[0]);
        console.log(form_data);
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            processData: false,
        }).done(function(data, textStatus, jqXHR){
            
            $('#result').text(data);



        }).fail(function(data){
            alert('error!');
        });
    });
}); 
