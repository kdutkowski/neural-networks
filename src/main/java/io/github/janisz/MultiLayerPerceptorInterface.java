package io.github.janisz;

import org.springframework.shell.core.CommandMarker;
import org.springframework.shell.core.annotation.CliAvailabilityIndicator;
import org.springframework.shell.core.annotation.CliCommand;
import org.springframework.stereotype.Component;

@Component
public class MultiLayerPerceptorInterface implements CommandMarker {


    @CliAvailabilityIndicator({"mlp"})
    public boolean isSimpleAvailable() {
        return true;
    }

    @CliCommand(value = "mlp", help = "Run mlp with default settings")
    public String simple() {
        return new MultiLayerPerceptor().buildSampleXorNetwork();
    }
}